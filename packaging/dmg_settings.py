# dmgbuild settings for the Avid Markup release .dmg.
#   python -m dmgbuild -s packaging/dmg_settings.py "Avid Markup" dist/AvidMarkup.dmg
# A simple drag-to-Applications layout.

import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Contents
files = [os.path.join(root, "dist", "AvidMarkup.app")]
symlinks = {"Applications": "/Applications"}

# Appearance
volume_name = "Avid Markup"
format = "UDZO"  # compressed
icon_locations = {
    "AvidMarkup.app": (140, 160),
    "Applications": (400, 160),
}
window_rect = ((200, 200), (560, 360))
default_view = "icon-view"
icon_size = 96
