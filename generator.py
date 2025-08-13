import os, re, json, hashlib, html, urllib.parse, xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import requests
from jinja2 import Environment, FileSystemLoader

# ===== CONFIG =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_KEY = os.getenv("UNSPLASH_KEY")

INTERESTS = [
    "911 theories",
    "false flags",
    "aliens",
    "comet 3i atlas",
    "ufos",
    "joe rogan podcasts",
    "president trump",
]

NEWS_LOOKBACK_HOURS = 24
ARTICLE_TARGET_WORDS = 800
NUM_PER_RUN = 4
SLIDER_LATEST = 4
HEADLINES_COUNT = 10
TRENDING_COUNT = 8
TICKER_COUNT = 12

DATA_FILE = "data/articles.json"
ART_DIR = "articles"
ASSETS_DIR = "assets"
INDEX_FILE = "index.html"
ARTICLE_TPL = "article.html.j2"
ARCHIVE_TPL = "articles_index.html.j2"

# ===== SETUP =====
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)
env = Environment(loader=FileSystemLoader("templates"))

# ===== HELPERS =====
def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "article"

def fingerprint(title: str, url: str) -> str:
    base = (title or "").strip().lower() + "|" + (url or "").strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def load_manifest() -> List[Dict]:
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # MIGRATION: backfill missing fingerprints
    changed = False
    for a in data:
        if "fingerprint" not in a:
            # prefer source_url/href if present, else fallback to file name
            url = a.get("source_url") or a.get("href") or a.get("file") or ""
            a["fingerprint"] = fingerprint(a.get("title", ""), url)
            changed = True
    if changed:
        save_manifest(data)
    return data

def save_manifest(items: List[Dict]):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

def google_news_rss(query: Optional[str] = None, limit: int = 20) -> List[Dict]:
    if query:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    else:
        url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        items = []
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub  = (item.findtext("pubDate") or "").strip()
            items.append({"title": title, "link": link, "pub": pub})
        return items
    except Exception as e:
        print("RSS error:", e)
        return []

def parse_rfc822(dt_str: str) -> Optional[datetime]:
    # Example: Tue, 13 Aug 2024 18:20:00 GMT
    try:
        return datetime.strptime(dt_str, "%a, %d %b %Y %H:%M:%S %Z")
    except Exception:
        try:
            return datetime.strptime(dt_str, "%a, %d %b %Y %H:%M:%S %z")
        except Exception:
            return None

def fetch_unsplash_image(query: str) -> str:
    if not UNSPLASH_KEY:
        return f"{ASSETS_DIR}/fallback-hero.jpg"
    try:
        r = requests.get(
            "https://api.unsplash.com/photos/random",
            headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"},
            params={"query": query, "orientation": "landscape", "content_filter": "high"},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        img_url = data.get("urls", {}).get("regular")
        if not img_url:
            return f"{ASSETS_DIR}/fallback-hero.jpg"
        img = requests.get(img_url, timeout=30)
        img.raise_for_status()
        fname = f"{slugify(query)}-{data.get('id','img')}.jpg"
        path = os.path.join(ASSETS_DIR, fname)
        with open(path, "wb") as f:
            f.write(img.content)
        return f"{ASSETS_DIR}/{fname}"
    except Exception as e:
        print("Unsplash error:", e)
        return f"{ASSETS_DIR}/fallback-hero.jpg"

def openai_generate_article(topic: str, source_title: str, source_url: str) -> str:
    # 800-word straight-news with skeptical, detail-obsessed crime-desk vibe
    prompt = (
        f"Write ~{ARTICLE_TARGET_WORDS} words in straight newspaper style (inverted pyramid). "
        f"Avoid slang; use neutral tone with careful attributions. Let subtle skepticism show via "
        f"sourcing and detail, like a meticulous crime-desk reporter. Use HTML paragraphs only "
        f"(<p>, optional <h2>, <blockquote> for quotes). Do not add HTML <html>/<body> wrappers.\n\n"
        f"Topic: {topic}\n"
        f"Source headline: {source_title}\n"
        f"Source URL: {source_url}\n"
    )
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a professional news writer."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        },
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def replace_block(html_text: str, start: str, end: str, inner: str) -> str:
    block = f"{start}\n{inner}\n{end}"
    return re.sub(re.escape(start) + r".*?" + re.escape(end), block, html_text, flags=re.S)

def toronto_weather() -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 43.65107,
        "longitude": -79.347015,
        "current": "temperature_2m,apparent_temperature,wind_speed_10m",
        "timezone": "America/Toronto",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        cur = r.json().get("current", {})
        t = cur.get("temperature_2m")
        feels = cur.get("apparent_temperature")
        wind = cur.get("wind_speed_10m")
        if t is None:
            return "Weather unavailable"
        return f"{t}°C (feels {feels}°C), wind {wind} km/h"
    except Exception as e:
        print("Weather error:", e)
        return "Weather unavailable"

# ===== MAIN =====
def main():
    manifest = load_manifest()
    seen = {a["fingerprint"] for a in manifest if a.get("fingerprint")}
    cutoff = datetime.utcnow() - timedelta(hours=NEWS_LOOKBACK_HOURS)

    created: List[Dict] = []
    # Pull candidates from each interest (up to a handful), then select
    candidates: List[Dict] = []
    for topic in INTERESTS:
        items = google_news_rss(topic, limit=8)
        for it in items:
            dt = parse_rfc822(it["pub"]) or datetime.utcnow()
            if dt < cutoff:
                continue
            candidates.append({"topic": topic, "title": it["title"], "link": it["link"], "date": dt})

    # Sort newest first; take unique by title+link and limit overall
    uniq = []
    seen_tmp = set()
    for c in sorted(candidates, key=lambda x: x["date"], reverse=True):
        fp = fingerprint(c["title"], c["link"])
        if fp in seen or fp in seen_tmp:
            continue
        uniq.append(c)
        seen_tmp.add(fp)
        if len(uniq) >= NUM_PER_RUN:
            break

    # Generate articles
    art_tpl = env.get_template(ARTICLE_TPL)
    for c in uniq:
        try:
            body_html = openai_generate_article(c["topic"], c["title"], c["link"])
        except Exception as e:
            print("OpenAI error:", e)
            continue
        img_path = fetch_unsplash_image(c["topic"])
        filename = f"{slugify(c['title'])}.html"
        page_html = art_tpl.render(
            title=c["title"],
            meta_description=c["title"],
            hero=f"../{img_path}",
            body=body_html,
            timestamp=c["date"].strftime("%B %d, %Y"),
        )
        with open(os.path.join(ART_DIR, filename), "w", encoding="utf-8") as f:
            f.write(page_html)
        fp = fingerprint(c["title"], c["link"])
        entry = {
            "title": c["title"],
            "href": f"articles/{filename}",
            "image": img_path,
            "date": c["date"].strftime("%Y-%m-%d %H:%M UTC"),
            "file": filename,
            "source_url": c["link"],
            "fingerprint": fp,
        }
        manifest.insert(0, entry)
        created.append(entry)
        seen.add(fp)

    if created:
        save_manifest(manifest)

    # Build homepage blocks from manifest + RSS/Weather
    latest = manifest[:SLIDER_LATEST]
    slides_html = "\n".join(
        [
            (
                "<a class='slide' href='articles/{file}' "
                "style=\"background-image:url('{img}')\">"
                "<div class='slide-content'><div class='kicker'>Top Story</div>"
                "<h2 class='slide-headline'>{ttl}</h2></div></a>"
            ).format(file=m["file"], img=m.get("image","assets/fallback-hero.jpg"), ttl=html.escape(m["title"]))
            for m in latest
        ]
    )

    headlines_src = manifest[:HEADLINES_COUNT]
    headlines_html = "\n".join([f"<li><a href=\"articles/{m['file']}\">{html.escape(m['title'])}</a></li>" for m in headlines_src])

    # Ticker: first N titles across interests
    ticker_titles: List[str] = []
    for topic in INTERESTS:
        for itm in google_news_rss(topic, limit=2):
            ticker_titles.append(itm["title"])
            if len(ticker_titles) >= TICKER_COUNT:
                break
        if len(ticker_titles) >= TICKER_COUNT:
            break
    ticker_text = " · ".join(ticker_titles) if ticker_titles else "Fresh updates every cycle."

    # Trending: top stories
    trending = google_news_rss(None, limit=TRENDING_COUNT)
    trending_html = "\n".join(
        [f"<li><a href=\"{t['link']}\" target=\"_blank\" rel=\"noopener\">{html.escape(t['title'])}</a></li>" for t in trending]
    ) or "<li>No data.</li>"

    weather_text = toronto_weather()

    # Update static index.html in-place by replacing marker blocks
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index_src = f.read()
    index_src = replace_block(index_src, "<!--SLIDES-->", "<!--/SLIDES-->", slides_html)
    index_src = replace_block(index_src, "<!--HEADLINES-->", "<!--/HEADLINES-->", headlines_html)
    index_src = replace_block(index_src, "<!--TICKER-->", "<!--/TICKER-->", ticker_text)
    index_src = replace_block(index_src, "<!--TRENDING-->", "<!--/TRENDING-->", trending_html)
    index_src = replace_block(index_src, "<!--WEATHER-->", "<!--/WEATHER-->", weather_text)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        f.write(index_src)

if __name__ == "__main__":
    main()
