#!/usr/bin/env python3
"""
Reset generated site content for Search 433.

What it does (safe by default):
- Deletes ALL files in /articles (the generated archive + article pages)
- Deletes ALL files in /assets (generated thumbnails/hero images)
- Re-creates empty /articles and /assets with .gitkeep
- Resets /data/articles.json to [] (empty)
- Optionally clears /data/story_memory.json (if --wipe-story-memory)
- Preserves: index.html, style.css, slider.js, generator.py, scraper.py,
  templates/, data/interests.txt, data/interests_weighted.json, update.yml, etc.

Usage examples:
  python reset_site.py
  python reset_site.py --keep-assets
  python reset_site.py --wipe-story-memory
"""

import argparse
import json
import os
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
ART_DIR = BASE / "articles"
ASSETS  = BASE / "assets"
DATA    = BASE / "data"

MANIFEST_JSON = DATA / "articles.json"
STORY_MEMORY  = DATA / "story_memory.json"

def rm_all_in_dir(path: Path):
    if not path.exists():
        return
    for p in path.iterdir():
        try:
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            print(f"[warn] Could not remove {p}: {e}")

def ensure_empty_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    # keep directory in git
    (path / ".gitkeep").write_text("", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-assets", action="store_true",
                    help="Do NOT delete images in /assets (default is to delete)")
    ap.add_argument("--wipe-story-memory", action="store_true",
                    help="Also clear /data/story_memory.json to {}")
    args = ap.parse_args()

    # 1) purge generated articles
    print("[reset] Clearing /articles …")
    rm_all_in_dir(ART_DIR)
    ensure_empty_dir(ART_DIR)

    # 2) purge generated images (unless kept)
    if args.keep_assets:
        print("[reset] Skipping /assets purge (--keep-assets set).")
        ASSETS.mkdir(parents=True, exist_ok=True)
    else:
        print("[reset] Clearing /assets …")
        rm_all_in_dir(ASSETS)
        ensure_empty_dir(ASSETS)

    # 3) reset manifest
    print("[reset] Resetting data/articles.json -> []")
    DATA.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text("[]\n", encoding="utf-8")

    # 4) optionally clear story memory
    if args.wipe_story_memory:
        if STORY_MEMORY.exists():
            print("[reset] Clearing data/story_memory.json -> {}")
            STORY_MEMORY.write_text("{}\n", encoding="utf-8")
        else:
            print("[reset] data/story_memory.json not present; nothing to clear.")

    print("[reset] Done.")

if __name__ == "__main__":
    main()
