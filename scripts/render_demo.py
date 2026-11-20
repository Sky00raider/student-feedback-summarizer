"""Run the real CLI on the sample survey and draw its output as a terminal-style SVG for the README.
Run: python scripts/render_demo.py   (writes docs/demo.svg; uses models/merged if present)"""
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
args = ["summarize", "samples/cs101_feedback.csv", "--digest"]
if (ROOT / "models" / "merged").is_dir():
    args += ["--merged", "models/merged"]
env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_VERBOSITY": "error"}
out = subprocess.run([sys.executable, "-m", "app.cli", *args], cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
                     check=True, env=env).stdout
shown = [a for a in args if a not in ("--merged", "models/merged")]
lines = [("cmd", "$ python -m app.cli " + " ".join(shown))]
for line in out.rstrip("\n").splitlines():
    kind = "head" if line.startswith("#") else ("neg" if " negative " in line[:22] else
                                                 "mix" if " mixed " in line[:22] else
                                                 "pos" if " positive " in line[:22] else "out")
    lines.append((kind, line.replace("**", "")))

COLORS = {"cmd": "#89b4fa", "head": "#f9e2af", "neg": "#f38ba8", "mix": "#fab387", "pos": "#a6e3a1", "out": "#cdd6f4"}
CW, LH, PAD, TOP = 8.4, 19, 18, 44
width = int(PAD * 2 + CW * max(len(t) for _, t in lines)) + 10
height = TOP + LH * len(lines) + PAD
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
       f'<rect width="{width}" height="{height}" rx="10" fill="#1e1e2e"/>',
       '<circle cx="20" cy="18" r="6" fill="#f38ba8"/><circle cx="40" cy="18" r="6" fill="#f9e2af"/>'
       '<circle cx="60" cy="18" r="6" fill="#a6e3a1"/>',
       '<g font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="14" xml:space="preserve">']
for i, (kind, text) in enumerate(lines):
    svg.append(f'<text x="{PAD}" y="{TOP + i * LH}" fill="{COLORS[kind]}">{escape(text)}</text>')
svg += ["</g>", "</svg>"]
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/demo.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
print(f"docs/demo.svg: {len(lines)} lines, {width}x{height}")
