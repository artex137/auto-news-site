import os, re, json, random, datetime, pathlib, unicodedata, hashlib, urllib.parse, xml.etree.ElementTree as ET
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

# Topics
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
HEADLINES_COUNT = 10
TRENDING_COUNT = 8
TICKER_COUNT = 12
FALLBACK = "assets/fallback-hero.jpg"

SYSTEM_STYLE = """You are a straight-news reporter. Write ~800 words (±10%) in newspaper style using the inverted pyramid:
lede (who/what/when/where/why/how) → nut graf → key facts with quotes & attributions → context → forward-looking kicker.
Neutral tone; AP-adjacent; modern vocabulary without slang. Attribute all claims to sources.

Return a JSON object with:
"title" (<= 90 chars),
"meta_description" (140–160 chars),
"body_html" (HTML only: <p>, <h2>, <ul>, <li>, <blockquote> for quotes),
"image_queries" (2–3 short phrases for related photos).
"""

URL_FINDER = """Return exactly one reputable, working news URL (no hard paywall if possible), published within the last 24 hours,
about the provided topic. Respond with ONLY the URL."""

ARTICLE_PROMPT = """Write the article described in the system message based strictly on this source URL:
{url}

Remember: ~800 words, with proper attributions and dates. Output JSON only as specified.
"""

# Writer style: we cannot include demeaning language about mental health. Use a serious crime-desk tone.
# (This replaces the earlier "criminally insane" phrasing with an intense, skeptical, detail-obsessed reporter voice.)

def slugify(value: str, max_len: int = 60) -> str:
    value = unicodedata.normalize('NFKD', value)
    value = re.sub(r'[^\w\s-]', '', value, flags=re.U).strip().lower()
    value = re.sub(r'[\s_-]+', '-', value)
    return value[:max_len].strip('-') or "article"

def fingerprint(url: str, title: str) -> str:
    norm = (url or "").strip().lower() + "|" + (title or "").strip().lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()

def ask_url(topic: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": URL_FINDER},
                  {"role": "user", "content": topic}],
        temperature=0.1, max_tokens=180
    )
    return r.choices[0].message.content.strip()

def write_article(url: str) -> Dict:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_STYLE},
                  {"role": "user", "content": ARTICLE_PROMPT.format(url=url)}],
        temperature=0.25, max_tokens=2200
    )
    text = r.choices[0].message.content.strip()
    m = re.search(r'\{.*\}\s*$', text, re.S)
    return json.loads(m.group(0) if m else text)

# ===== Unsplash (uses existing UNSPLASH_KEY) =====
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

# ===== RSS utilities (Google News) =====
def google_news_rss(query: Optional[str]=None, limit:int=20) -> List[Dict]:
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
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            items.append({"title": title.strip(), "link": link.strip()})
        return items
    except Exception as e:
        print("RSS error:", e)
        return []

# ===== Weather (Open-Meteo, no key) =====
def toronto_weather() -> str:
    # Toronto approx lat/long
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 43.65107,
        "longitude": -79.347015,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min",
        "timezone": "America/Toronto"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        cur = data.get("current", {})
        daily = data.get("daily", {})
        t = cur.get("temperature_2m")
        feels = cur.get("apparent_temperature")
        wind = cur.get("wind_speed_10m")
        tmax = daily.get("temperature_2m_max", ["?"])[0]
        tmin = daily.get("temperature_2m_min", ["?"])[0]
        return f"{t}°C (feels {feels}°C), wind {wind} km/h · Today {tmin}°/{tmax}°"
    except Exception as e:
        print("Weather error:", e)
        return "Weather unavailable"

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

def update_index(slides_html: str, headlines_html: str, ticker_text: str, trending_html: str, weather_text: str):
    html = INDEX.read_text(encoding="utf-8")
    html = replace_block(html, "<!--SLIDES-->", "<!--/SLIDES-->", slides_html)
    html = replace_block(html, "<!--HEADLINES-->", "<!--/HEADLINES-->", headlines_html)
    html = replace_block(html, "<!--TICKER-->", "<!--/TICKER-->", ticker_text)
    html = replace_block(html, "<!--TRENDING-->", "<!--/TRENDING-->", trending_html)
    html = replace_block(html, "<!--WEATHER-->", "<!--/WEATHER-->", weather_text)
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

# ===== Main =====
def main():
    ART_DIR.mkdir(exist_ok=True, parents=True)
    manifest = load_manifest()
    seen_fps = {m.get("fp") for m in manifest if m.get("fp")}

    created: List[Dict] = []
    topics = random.sample(INTERESTS, k=min(NUM_ARTICLES, len(INTERESTS)))

    for topic in topics:
        try:
            source_url = ask_url(topic)
            payload = write_article(source_url)
            title = (payload.get("title") or "Untitled").strip()
            fp = fingerprint(source_url, title)
            if fp in seen_fps:
                print("Duplicate skipped:", title)
                continue

            slug = slugify(title)
            ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

            queries = payload.get("image_queries") or [title, topic]
            hero_repo_rel = get_story_image(queries, slug)
            hero_rel_for_article = "../" + hero_repo_rel if hero_repo_rel.startswith("assets/") else "../" + hero_repo_rel.lstrip("./")

            html = render_article(payload, hero_rel_for_article, ts)
            filename = f"{slug}.html"
            (ART_DIR / filename).write_text(html, encoding="utf-8")

            item = {"title": title, "file": filename, "image": hero_repo_rel, "date": ts, "source_url": source_url, "fp": fp}
            manifest.append(item)
            created.append(item)
            seen_fps.add(fp)
        except Exception as e:
            print("Error generating:", e)

    if created:
        save_manifest(manifest)
        render_archive(manifest)

    latest = sorted(manifest, key=lambda x: x["date"], reverse=True)[:SLIDER_LATEST]
    slides = [
        f"<a class=\\"slide\\" href=\\"articles/{m['file']}\\\" style=\\"background-image:url('{m['image']}')\\">"
        f"<div class=\\"slide-content\\"><div class='kicker'>Top Story</div><h2 class=\\"slide-headline\\">{m['title']}</h2></div></a>"
        for m in latest
    ]
    headlines_src = sorted(manifest, key=lambda x: x["date"], reverse=True)[:HEADLINES_COUNT]
    headlines = [f"<li><a href=\\"articles/{m['file']}\\">{m['title']}</a></li>" for m in headlines_src]

    ticker_items: List[str] = []
    for t in INTERESTS:
        for itm in google_news_rss(t, limit=2):
            ticker_items.append(itm["title"])
            if len(ticker_items) >= TICKER_COUNT: break
        if len(ticker_items) >= TICKER_COUNT: break
    ticker_text = " · ".join(ticker_items) if ticker_items else "Fresh updates every cycle."

    trending_feed = google_news_rss(None, limit=TRENDING_COUNT)
    trending_html = "\\n".join([f"<li><a href=\\"{i['link']}\\\" target=\\"_blank\\" rel=\\"noopener\\">{i['title']}</a></li>" for i in trending_feed]) or "<li>No data.</li>"

    weather_text = toronto_weather()

    update_index("\\n".join(slides), "\\n".join(headlines), ticker_text, trending_html, weather_text)

if __name__ == "__main__":
    main()
