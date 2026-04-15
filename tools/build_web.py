#!/usr/bin/env python3
"""
Build the travel-py wheel and deploy it into docs/ for GitHub Pages.

Usage:
    uv run python tools/build_web.py

What it does:
  1. Runs `uv build --wheel` to produce a pure-Python wheel.
  2. Copies the wheel into docs/ (removes any previous wheel for this package).
  3. Writes docs/wheel_info.json so interactive.html can discover the filename
     without hard-coding a version string.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def main() -> None:
    DOCS.mkdir(exist_ok=True)

    # ── 1. Build wheel ────────────────────────────────────────────────────
    tmp_out = ROOT / "_wheel_build_tmp"
    tmp_out.mkdir(exist_ok=True)
    try:
        result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(tmp_out)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="")
    except subprocess.CalledProcessError as e:
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

    # ── 2. Find the built wheel ───────────────────────────────────────────
    wheels = sorted(tmp_out.glob("travel_py-*.whl"))
    if not wheels:
        sys.exit("ERROR: no wheel found after build")
    wheel_src = wheels[-1]  # highest version if multiple

    # ── 3. Remove old wheels in docs/, copy new one ───────────────────────
    for old in DOCS.glob("travel_py-*.whl"):
        old.unlink()
        print(f"Removed old wheel: {old.name}")

    wheel_dst = DOCS / wheel_src.name
    shutil.copy2(wheel_src, wheel_dst)
    print(f"Copied wheel → docs/{wheel_src.name}")

    # ── 4. Write manifest ─────────────────────────────────────────────────
    manifest = {"url": wheel_src.name}
    manifest_path = DOCS / "wheel_info.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote docs/wheel_info.json: {manifest}")

    # ── 5. Cleanup ────────────────────────────────────────────────────────
    shutil.rmtree(tmp_out)


if __name__ == "__main__":
    main()
