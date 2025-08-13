import os, re, json, random, datetime, pathlib, unicodedata
from typing import List, Dict, Optional
import requests
from openai import OpenAI
from jinja2 import Template

# ===== Config =====
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE = pathlib.Path(__file__).parent.resolve()
ART_DIR = BASE / "articles"
ASSETS = BASE / "assets"
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MANIFEST = DATA_DIR / "articles.json"

TPL_DIR = BASE / "templates"
ARTICLE_TPL = TPL_DIR / "article.html.j2"
ARCHIVE_TPL = TPL_DIR / "articles_index.html.j2"
INDEX = BASE / "index.html"

# Topics (your new list)
INTERESTS = [
    "911 theories",
    "false flags",
    "aliens",
    "comet 3i atlas",
    "ufos",
    "joe rogan podcasts",
    "president trump"
]

NUM_ARTICLES = 4
SLIDER_LATEST = 4
HEADLINES_COUNT = 8
FALLBACK = "assets/fallback-hero.jpg"

SYSTEM_STYLE = """Write in straight newspaper style (inverted pyramid). Neutral tone, clear sourcing, quotes with attribution.
Return JSON with: "title", "meta_description", "body_html", "image_queries" (2-3 concise)."""

URL_FINDER = """Return exactly one reputable, working news URL (no paywall if possible) less than 6 hours old
about the provided topic. Respond with ONLY the URL."""

ARTICLE_PROMPT = """Write an article per the system style based on:
{url}

Return JSON:
- "title": headline (<= 90 chars)
- "meta_description": 140–160 chars
- "body_html": HTML body with <p>, <h2>, <blockquote> (no <html>/<body> wrappers)
- "image_queries": 2–3 short phrases for related photos"""

# ===== Helpers =====
def slugify(value: str, max_len: int = 60) -> str:
    value = unicodedata.normalize('NFKD', value)
    value = re.sub(r'[^\w\s-]', '', value, flags=re.U).strip().lower()
    value = re.sub(r'[\s_-]+', '-', value)
    return value[:max_len].strip('-') or "article"

def ask_url(topic: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": URL_FINDER},
                  {"role": "user", "content": topic}],
        temperature=0.2, max_tokens=200
    )
    return r.choices[0].message.content.strip()

def write_article(url: str) -> Dict:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_STYLE},
                  {"role": "user", "content": ARTICLE_PROMPT.format(url=url)}],
        temperature=0.3, max_tokens=1600
    )
    text = r.choices[0].message.content.strip()
    m = re.search(r'\{.*\}\s*$', text, re.S)
    return json.loads(m.group(0) if m else text)

# ===== Unsplash =====
UNSPLASH_KEY = os.getenv("UNSPLASH_KEY")
def unsplash_image(query: str) -> Optional[str]:
    if not UNSPLASH_KEY: return None
    try:
        r = requests.get(
            "https://api.unsplash.com/photos/random",
            headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"},
            params={"query": query, "orientation": "landscape"},
            timeout=20
        )
        r.raise_for_status()
        data = r.json()
        return data.get("urls", {}).get("regular")
    except Exception as e:
        print("Unsplash error:", e)
        return None

def download_image(url: str, filename_hint: str) -> str:
    try:
        ASSETS.mkdir(exist_ok=True, parents=True)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        ext = ".jpg"
        if "image/png" in resp.headers.get("Content-Type",""):
            ext = ".png"
        fname = f"{filename_hint}{ext}"
        path = ASSETS / fname
        with open(path, "wb") as f:
            f.write(resp.content)
        return f"assets/{fname}"
    except Exception as e:
        print("Image download error:", e)
        return FALLBACK if (BASE / FALLBACK).exists() else ""

def get_story_image(queries: List[str], title_slug: str) -> str:
    for q in queries[:3]:
        url = unsplash_image(q)
        if url:
            return download_image(url, f"{title_slug}")
    return FALLBACK if (BASE / FALLBACK).exists() else ""

# ===== Rendering & index injections =====
def render_article(payload: Dict, hero_rel_for_article: str, ts_utc: str) -> str:
    html = ARTICLE_TPL.read_text(encoding="utf-8")
    tpl = Template(html)
    return tpl.render(
        title=payload.get("title", "Untitled"),
        meta_description=payload.get("meta_description", ""),
        body=payload.get("body_html", ""),
        hero=hero_rel_for_article,
        images=[],
        timestamp=ts_utc
    )

def replace_block(html: str, start_tag: str, end_tag: str, inner: str) -> str:
    block = f"{start_tag}\n{inner}\n{end_tag}"
    return re.sub(re.escape(start_tag) + r".*?" + re.escape(end_tag), block, html, flags=re.S)

def update_index(slides_html: str, headlines_html: str):
    html = INDEX.read_text(encoding="utf-8")
    html = replace_block(html, "<!--SLIDES-->", "<!--/SLIDES-->", slides_html)
    html = replace_block(html, "<!--HEADLINES-->", "<!--/HEADLINES-->", headlines_html)
    INDEX.write_text(html, encoding="utf-8")

def render_archive(manifest: List[Dict]):
    tpl = Template(ARCHIVE_TPL.read_text(encoding="utf-8"))
    articles = [{
        "href": f"./{m['file']}",
        "title": m["title"],
        "image": m.get("image", FALLBACK),
        "date": m["date"]
    } for m in sorted(manifest, key=lambda x: x["date"], reverse=True)]
    out = tpl.render(articles=articles)
    ART_DIR.mkdir(exist_ok=True, parents=True)
    (ART_DIR / "index.html").write_text(out, encoding="utf-8")

def load_manifest() -> List[Dict]:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_manifest(items: List[Dict]):
    MANIFEST.write_text(json.dumps(items, indent=2), encoding="utf-8")

# ===== Main (with duplicate protection) =====
def main():
    ART_DIR.mkdir(exist_ok=True, parents=True)
    manifest = load_manifest()
    seen_urls = {m.get("source_url","") for m in manifest if m.get("source_url")}
    seen_titles = {m["title"] for m in manifest}

    created: List[Dict] = []
    topics = random.sample(INTERESTS, k=min(NUM_ARTICLES, len(INTERESTS)))

    for topic in topics:
        try:
            url = ask_url(topic)
            if url in seen_urls:
                print("Duplicate URL skipped:", url)
                continue

            payload = write_article(url)
            title = (payload.get("title") or "Untitled").strip()
            if title in seen_titles:
                print("Duplicate title skipped:", title)
                continue

            slug = slugify(title)
            ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

            queries = payload.get("image_queries") or [title, topic]
            hero_repo_rel = get_story_image(queries, slug)
            hero_rel_for_article = "../" + hero_repo_rel if hero_repo_rel.startswith("assets/") else "../" + hero_repo_rel.lstrip("./")

            html = render_article(payload, hero_rel_for_article, ts)
            filename = f"{slug}.html"
            (ART_DIR / filename).write_text(html, encoding="utf-8")

            item = {"title": title, "file": filename, "image": hero_repo_rel, "date": ts, "source_url": url}
            manifest.append(item)
            created.append(item)
            seen_urls.add(url); seen_titles.add(title)

        except Exception as e:
            print("Error generating:", e)

    if created:
        save_manifest(manifest)
        render_archive(manifest)

        latest = sorted(manifest, key=lambda x: x["date"], reverse=True)[:SLIDER_LATEST]
        slides = [
            f"<a class=\"slide\" href=\"articles/{m['file']}\" style=\"background-image:url('{m['image']}')\">"
            f"<div class=\"slide-content\"><h2 class=\"slide-headline\">{m['title']}</h2></div></a>"
            for m in latest
        ]

        # Build right-rail headlines
        headlines_src = sorted(manifest, key=lambda x: x["date"], reverse=True)[:HEADLINES_COUNT]
        headlines = [f"<li><a href=\"articles/{m['file']}\">{m['title']}</a></li>" for m in headlines_src]

        update_index("\n".join(slides), "\n".join(headlines))

if __name__ == "__main__":
    main()
