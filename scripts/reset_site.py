#!/usr/bin/env python3
"""
Reset the site to a clean testing state by removing all generated content
and restoring empty data files.

Actions:
- Delete all generated article pages in /articles (except we recreate a minimal index)
- Delete all images in /assets
- Reset /data/articles.json to []
- Reset /data/story_memory.json to {}
- Ensure /articles and /assets exist with a .gitkeep
- Write a minimal /articles/index.html to avoid broken nav

Idempotent: safe to run multiple times.
"""

from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

ARTICLES_DIR = REPO_ROOT / "articles"
ASSETS_DIR = REPO_ROOT / "assets"
DATA_DIR = REPO_ROOT / "data"

ARTICLES_JSON = DATA_DIR / "articles.json"
STORY_MEMORY_JSON = DATA_DIR / "story_memory.json"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def write_gitkeep(path: Path):
    keep = path / ".gitkeep"
    if not keep.exists():
        keep.write_text("", encoding="utf-8")

def purge_directory(path: Path, delete_everything=True, keep_files=None):
    """
    Delete everything in a directory except any names listed in keep_files.
    If delete_everything=True and keep_files is None/empty, remove all files.
    """
    keep_files = set(keep_files or [])
    if not path.exists():
        return 0

    removed = 0
    for p in path.iterdir():
        # Skip .gitkeep (we recreate it later)
        if p.name == ".gitkeep":
            continue

        if p.is_dir():
            # Remove subfolders entirely
            for sub in p.rglob("*"):
                if sub.is_file():
                    try:
                        sub.unlink()
                        removed += 1
                    except FileNotFoundError:
                        pass
            try:
                p.rmdir()
            except OSError:
                # If non-empty for any reason, ignore
                pass
            continue

        # Files
        if delete_everything and p.name not in keep_files:
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
    return removed

def write_minimal_articles_index():
    """
    Minimal /articles/index.html that uses existing style.css and keeps brand/header structure.
    Assumes style.css and site header markup live at top-level like index.html.
    """
    target = ARTICLES_DIR / "index.html"
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Articles — Search 433</title>
  <meta name="description" content="No articles yet.">
  <link rel="stylesheet" href="../style.css">
</head>
<body class="theme-darkblue">
  <header class="site-header">
    <div class="container row between middle">
      <a class="brand" href="../index.html" aria-label="Search 433">
        <svg class="logo-svg" viewBox="0 0 160 40" width="160" height="40" aria-hidden="true">
          <defs>
            <linearGradient id="g433" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#7fb2ff"/>
              <stop offset="100%" stop-color="#2f6fff"/>
            </linearGradient>
          </defs>
          <rect x="0" y="0" rx="10" ry="10" width="52" height="40" fill="url(#g433)"></rect>
          <text x="26" y="26" text-anchor="middle" font-family="Inter, Arial" font-size="18" font-weight="800" fill="#fff">433</text>
          <text x="64" y="26" font-family="DM Serif Display, serif" font-size="20" fill="#e9f0ff"><tspan font-style="italic">Search</tspan> 433</text>
        </svg>
      </a>
      <nav class="nav">
        <a href="../index.html" class="nav-link">Home</a>
        <a href="./index.html" class="nav-link active">Articles</a>
      </nav>
    </div>
  </header>

  <main class="section">
    <div class="container">
      <h1 class="section-title">Articles</h1>
      <div class="archive">
        <p class="muted" style="margin-top:1rem">No articles yet. Use your generator to create fresh content.</p>
      </div>
    </div>
  </main>

  <footer class="site-footer">
    <div class="container row between wrap">
      <p>© <script>document.write(new Date().getFullYear())</script> Search 433</p>
      <p class="muted">Clean slate.</p>
    </div>
  </footer>
</body>
</html>
"""
    target.write_text(html, encoding="utf-8")

def reset_data_files():
    ensure_dir(DATA_DIR)

    # articles.json -> []
    try:
        ARTICLES_JSON.write_text(json.dumps([], indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not write {ARTICLES_JSON}: {e}", file=sys.stderr)

    # story_memory.json -> {}
    try:
        STORY_MEMORY_JSON.write_text(json.dumps({}, indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not write {STORY_MEMORY_JSON}: {e}", file=sys.stderr)

def main():
    ensure_dir(ARTICLES_DIR)
    ensure_dir(ASSETS_DIR)

    # Purge all generated HTML articles in /articles and everything in /assets
    removed_articles = purge_directory(ARTICLES_DIR, delete_everything=True, keep_files=set())
    removed_assets = purge_directory(ASSETS_DIR, delete_everything=True, keep_files=set())

    # Recreate minimal articles index and .gitkeep sentinels
    write_minimal_articles_index()
    write_gitkeep(ARTICLES_DIR)
    write_gitkeep(ASSETS_DIR)

    # Reset data
    reset_data_files()

    print("=== Reset Summary ===")
    print(f"Removed files in /articles: {removed_articles}")
    print(f"Removed files in /assets:   {removed_assets}")
    print(f"Reset: {ARTICLES_JSON.relative_to(REPO_ROOT)} to []")
    print(f"Reset: {STORY_MEMORY_JSON.relative_to(REPO_ROOT)} to {{}}")
    print("Site is now clean. Generate fresh articles when ready.")

if __name__ == "__main__":
    main()
