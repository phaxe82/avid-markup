const AVID_COLORS = ["Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "White", "Black"];
const COLOR_CSS = {
  Red: "#e06c75", Green: "#67c073", Blue: "#5a9bf2", Cyan: "#4fd2d2",
  Magenta: "#cf6fd6", Yellow: "#f2c14e", White: "#e7eaf0", Black: "#0c0d10",
};

const state = {
  audioUrl: null,
  jobId: null,
  rawSegments: [],  // unmodified fragments from the server
  segments: [],     // {start, end, text, speaker, include} after grouping/editing
  groupMode: "turn",
  speakers: [],     // {speaker, start, end, text}
  labels: {},       // SPEAKER_00 -> "Sam"
  excludedSpeakers: {}, // SPEAKER_02 -> true (drop all its lines, e.g. crew)
  pollTimer: null,
  originalName: "scene",
};

const $ = (id) => document.getElementById(id);
const player = $("player");

function fmtTc(seconds) {
  const f = Math.round((seconds % 1) * 25);
  const s = Math.floor(seconds);
  const hh = String(Math.floor(s / 3600)).padStart(2, "0");
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const ss = String(s % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}:${String(f).padStart(2, "0")}`;
}

// ---- audio sample playback ----
let stopAt = null;
player.addEventListener("timeupdate", () => {
  if (stopAt !== null && player.currentTime >= stopAt) {
    player.pause();
    stopAt = null;
  }
});
function playClip(start, end) {
  if (!state.audioUrl) return;
  stopAt = end;
  player.currentTime = start;
  player.play();
}

// ---- init ----
async function init() {
  const cfg = await (await fetch("/api/config")).json();
  $("model-size").value = cfg.model_size || "small";
  const note = $("diar-note");
  if (cfg.diarization_enabled) {
    note.textContent = "Speaker diarization is on — each voice will be separated automatically.";
  } else {
    note.textContent = "No HuggingFace token set, so diarization is OFF — you'll get one unlabelled speaker. See the README to enable it.";
    note.classList.add("warn");
  }
  if (!cfg.llm_available) {
    const llm = $("llm-correct");
    llm.checked = false;
    llm.disabled = true;
    llm.closest("label").title = "Install the ML extra (mlx-lm) to enable local AI refinement.";
    const trim = $("trim-btn");
    trim.disabled = true;
    trim.title = "Install the ML extra (mlx-lm) to enable local AI trimming.";
  }
  const sel = $("single-color");
  AVID_COLORS.forEach((c) => {
    const o = document.createElement("option");
    o.value = c; o.textContent = c; if (c === "Yellow") o.selected = true;
    sel.appendChild(o);
  });
  $("color-mode").addEventListener("change", () => {
    $("single-color-wrap").style.display =
      $("color-mode").value === "per_speaker" ? "none" : "";
  });
  $("group-mode").addEventListener("change", () => {
    state.groupMode = $("group-mode").value;
    applyGrouping();
    setStatus(`${state.segments.length} markers (${state.groupMode}).`);
  });
}

// ---- timecode input masking ----
// User types digits only; colons are inserted automatically.
$("start-tc").addEventListener("input", (e) => {
  const digits = e.target.value.replace(/\D/g, "").slice(0, 8);
  const parts = [digits.slice(0, 2), digits.slice(2, 4), digits.slice(4, 6), digits.slice(6, 8)];
  e.target.value = parts.filter((p) => p.length).join(":");
});

// ---- timecode extraction from filename ----
// Looks for an 8-digit run (HHMMSSFF) or separated HH_MM_SS_FF / HH-MM-SS-FF in the filename.
function extractTcFromFilename(name) {
  const run = name.match(/(\d{8})/);
  if (run) {
    const d = run[1];
    const mm = d.slice(2, 4), ss = d.slice(4, 6), ff = d.slice(6, 8);
    if (+mm < 60 && +ss < 60 && +ff < 25)
      return `${d.slice(0, 2)}:${mm}:${ss}:${ff}`;
  }
  const sep = name.match(/(\d{2})[_\-](\d{2})[_\-](\d{2})[_\-](\d{2})/);
  if (sep) {
    const [, hh, mm, ss, ff] = sep;
    if (+mm < 60 && +ss < 60 && +ff < 25) return `${hh}:${mm}:${ss}:${ff}`;
  }
  return null;
}

$("file").addEventListener("change", () => {
  const file = $("file").files[0];
  if (!file) return;
  const tc = extractTcFromFilename(file.name);
  if (tc) $("start-tc").value = tc;
});

// ---- transcription ----
$("transcribe-btn").addEventListener("click", async () => {
  const fileInput = $("file");
  if (!fileInput.files.length) { setStatus("Choose an audio file first.", true); return; }
  const file = fileInput.files[0];
  state.originalName = file.name.replace(/\.[^.]+$/, "");

  const fd = new FormData();
  fd.append("file", file);
  fd.append("model_size", $("model-size").value);
  fd.append("language", $("language").value);
  fd.append("llm_correct", $("llm-correct").checked ? "true" : "false");
  if ($("min-speakers").value) fd.append("min_speakers", $("min-speakers").value);
  if ($("max-speakers").value) fd.append("max_speakers", $("max-speakers").value);

  $("transcribe-btn").disabled = true;
  setStatus("Uploading…");
  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    state.jobId = data.job_id;
    state.audioUrl = data.audio_url;
    player.src = data.audio_url;
    setStatus("Transcribing… this runs locally and can take a little while.");
    poll();
  } catch (e) {
    setStatus(`Error: ${e.message}`, true);
    $("transcribe-btn").disabled = false;
  }
});

function poll() {
  state.pollTimer = setInterval(async () => {
    const job = await (await fetch(`/api/jobs/${state.jobId}`)).json();
    if (job.status === "done") {
      clearInterval(state.pollTimer);
      onResult(job.result);
    } else if (job.status === "error") {
      clearInterval(state.pollTimer);
      setStatus(`Transcription failed: ${job.error}`, true);
      $("transcribe-btn").disabled = false;
    } else if (job.stage) {
      setStatus(job.stage);
    }
  }, 2000);
}

function onResult(result) {
  state.rawSegments = result.segments;
  state.speakers = result.speakers;
  state.labels = {};
  state.excludedSpeakers = {};
  state.speakers.forEach((s) => { state.labels[s.speaker] = ""; });
  applyGrouping();
  const corrected = result.corrected_speakers || 0;
  const refined = corrected
    ? ` AI refined ${corrected} speaker label${corrected === 1 ? "" : "s"}.`
    : "";
  setStatus(`Done — ${state.segments.length} markers, ${state.speakers.length} speaker(s).${refined}`);
  $("transcribe-btn").disabled = false;
  renderSpeakers();
  $("step-speakers").classList.toggle("hidden", state.speakers.length === 0);
  $("step-transcript").classList.remove("hidden");
  $("step-output").classList.remove("hidden");
}

// ---- speakers ----
function speakerColor(speakerId) {
  const idx = state.speakers.findIndex((s) => s.speaker === speakerId);
  return AVID_COLORS[(idx < 0 ? 0 : idx) % AVID_COLORS.length];
}

function renderSpeakers() {
  const list = $("speaker-list");
  list.innerHTML = "";
  state.speakers.forEach((sp) => {
    const card = document.createElement("div");
    card.className = "speaker-card";

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = COLOR_CSS[speakerColor(sp.speaker)];

    const orig = document.createElement("span");
    orig.className = "orig";
    orig.textContent = sp.speaker;

    const play = document.createElement("button");
    play.className = "play";
    play.textContent = "▶ Sample";
    play.addEventListener("click", () => playClip(sp.start, sp.end));

    const sample = document.createElement("span");
    sample.className = "sample";
    sample.textContent = `"${sp.text}"`;

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Name (e.g. Sam)";
    input.value = state.labels[sp.speaker] || "";
    input.addEventListener("input", () => {
      state.labels[sp.speaker] = input.value;
      renderSegments();
    });
    // Committing a name can make two speakers share an identity — re-group so
    // their now-adjacent turns merge into one marker.
    input.addEventListener("change", () => applyGrouping());

    // Exclude a whole speaker (e.g. off-camera crew) from the export in one click.
    const exWrap = document.createElement("label");
    exWrap.className = "exclude-toggle";
    const ex = document.createElement("input");
    ex.type = "checkbox";
    ex.checked = !!state.excludedSpeakers[sp.speaker];
    ex.addEventListener("change", () => {
      state.excludedSpeakers[sp.speaker] = ex.checked;
      state.segments.forEach((seg) => {
        if (seg.speaker === sp.speaker) seg.include = !ex.checked;
      });
      card.classList.toggle("excluded", ex.checked);
      renderSegments();
    });
    exWrap.append(ex, document.createTextNode("Exclude (crew)"));
    card.classList.toggle("excluded", ex.checked);

    card.append(swatch, orig, play, sample, input, exWrap);
    list.appendChild(card);
  });
}

// ---- grouping ----
function endsSentence(text) {
  return /[.!?]["')\]]?\s*$/.test(text);
}

// Two consecutive lines count as the same speaker if they resolve to the same
// name. Diarization often splits one person across several SPEAKER_xx ids; once
// they're all labelled "Nadine" we treat them as one so a turn merges into a
// single marker. Unlabelled speakers fall back to their raw id (so distinct
// unnamed speakers still don't merge).
function speakerKey(speakerId) {
  return (state.labels[speakerId] || "").trim() || speakerId;
}

// Merge the raw WhisperX fragments into fuller markers so a sentence is never
// split across markers. "sentence" joins same-speaker fragments until a sentence
// ends; "turn" joins a whole speaker turn; "raw" keeps fragments as-is.
function groupSegments(raw, mode) {
  if (mode === "raw") {
    return raw.map((s) => ({ ...s, include: true }));
  }
  const out = [];
  let cur = null;
  for (const seg of raw) {
    const sameSpeaker = cur && speakerKey(cur.speaker) === speakerKey(seg.speaker);
    const keepBuilding = mode === "turn" ? sameSpeaker : sameSpeaker && !endsSentence(cur.text);
    if (keepBuilding) {
      cur.text = `${cur.text} ${seg.text}`.replace(/\s+/g, " ").trim();
      cur.end = seg.end;
    } else {
      if (cur) out.push(cur);
      cur = { start: seg.start, end: seg.end, text: seg.text, speaker: seg.speaker, include: true };
    }
  }
  if (cur) out.push(cur);
  return out;
}

function applyGrouping() {
  state.segments = groupSegments(state.rawSegments, state.groupMode);
  // re-apply whole-speaker exclusions (kept by speaker id across regroupings)
  state.segments.forEach((seg) => {
    if (state.excludedSpeakers[seg.speaker]) seg.include = false;
  });
  renderSegments();
}

// ---- segments ----
function renderSegments() {
  const list = $("segment-list");
  list.innerHTML = "";
  state.segments.forEach((seg, i) => {
    const row = document.createElement("div");
    row.className = "seg-row" + (seg.include ? "" : " excluded");

    const tc = document.createElement("span");
    tc.className = "col-tc";
    const play = document.createElement("button");
    play.className = "play tc-play";
    play.textContent = "▶";
    play.addEventListener("click", () => playClip(seg.start, seg.end));
    const tcText = document.createElement("span");
    tcText.textContent = fmtTc(seg.start);
    tc.append(play, tcText);

    const spk = document.createElement("select");
    spk.className = "col-spk";
    const optNone = document.createElement("option");
    optNone.value = ""; optNone.textContent = "(none)";
    spk.appendChild(optNone);
    state.speakers.forEach((sp) => {
      const o = document.createElement("option");
      o.value = sp.speaker;
      o.textContent = state.labels[sp.speaker] || sp.speaker;
      if (sp.speaker === seg.speaker) o.selected = true;
      spk.appendChild(o);
    });
    spk.addEventListener("change", () => { seg.speaker = spk.value; });

    const txt = document.createElement("input");
    txt.type = "text"; txt.className = "col-txt"; txt.value = seg.text;
    txt.addEventListener("input", () => { seg.text = txt.value; });

    const incWrap = document.createElement("span");
    incWrap.className = "col-inc";
    const inc = document.createElement("input");
    inc.type = "checkbox"; inc.checked = seg.include;
    inc.addEventListener("change", () => {
      seg.include = inc.checked;
      row.classList.toggle("excluded", !seg.include);
    });
    incWrap.appendChild(inc);

    row.append(tc, spk, txt, incWrap);
    list.appendChild(row);
  });
}

// ---- export ----
$("export-btn").addEventListener("click", () => {
  const startTc = $("start-tc").value.trim();
  if (!/^\d{1,2}:\d{2}:\d{2}:\d{2}$/.test(startTc)) {
    setExportStatus("Start timecode must be HH:MM:SS:FF (e.g. 10:00:00:00).", true);
    return;
  }
  const body = {
    start_tc: startTc,
    fps: 25,
    author: $("author").value.trim() || "Transcriber",
    track: $("track").value.trim() || "V1",
    label_separator: " - ",
    speaker_labels: state.labels,
    color_mode: $("color-mode").value,
    single_color: $("single-color").value,
    duration_mode: $("duration-mode").value,
    filename: state.originalName,
    segments: state.segments.map((s) => ({
      start: s.start, end: s.end, text: s.text, speaker: s.speaker, include: s.include,
    })),
  };
  // Prepare the file server-side, then navigate to a GET URL that ENDS in the
  // filename. That guarantees the saved name in every browser — Safari names
  // blob/Content-Disposition downloads with a UUID and no extension otherwise.
  $("export-btn").disabled = true;
  setExportStatus("Building…");
  (async () => {
    try {
      const res = await fetch("/api/export_prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const { download_url, filename } = await res.json();
      window.location.href = download_url;
      setExportStatus(`Downloaded ${filename}`);
    } catch (e) {
      setExportStatus(`Error: ${e.message}`, true);
    } finally {
      $("export-btn").disabled = false;
    }
  })();
});

// ---- AI trim (drop low-value lines) ----
// Sends only the currently-kept lines to the local LLM; it returns the indices
// (into that submitted list) to drop, which we map back and un-tick. It never
// re-includes a line, so manual keeps and crew exclusions are preserved.
$("trim-btn").addEventListener("click", async () => {
  const items = [];
  state.segments.forEach((seg, i) => {
    if (seg.include) items.push({ i, speaker: seg.speaker || "", text: seg.text });
  });
  if (!items.length) { setTrimStatus("No lines left to trim.", true); return; }

  $("trim-btn").disabled = true;
  setTrimStatus("Trimming… reading the transcript locally.");
  try {
    const res = await fetch("/api/triage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        level: $("trim-level").value,
        guidance: $("trim-guidance").value.trim(),
        segments: items.map((it) => ({ speaker: it.speaker, text: it.text })),
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const { drop, kept, dropped } = await res.json();
    drop.forEach((d) => {
      const target = items[d];
      if (target) state.segments[target.i].include = false;
    });
    renderSegments();
    setTrimStatus(
      `Trimmed ${dropped} low-value line${dropped === 1 ? "" : "s"} — ` +
      `${kept} marker${kept === 1 ? "" : "s"} kept. Review the unticked rows.`
    );
  } catch (e) {
    setTrimStatus(`Error: ${e.message}`, true);
  } finally {
    $("trim-btn").disabled = false;
  }
});

$("reinclude-btn").addEventListener("click", () => {
  state.segments.forEach((seg) => { seg.include = true; });
  renderSegments();
  setTrimStatus("All lines re-included.");
});

function setStatus(msg, isError = false) {
  const el = $("status");
  el.textContent = msg;
  el.classList.toggle("error", isError);
}
function setExportStatus(msg, isError = false) {
  const el = $("export-status");
  el.textContent = msg;
  el.classList.toggle("error", isError);
}
function setTrimStatus(msg, isError = false) {
  const el = $("trim-status");
  el.textContent = msg;
  el.classList.toggle("error", isError);
}

init();
