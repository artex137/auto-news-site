import os, re, json, hashlib, html, urllib.parse, xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from jinja2 import Template

# ===== Config =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_KEY = os.getenv("UNSPLASH_KEY")

BASE = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE, "articles")
ASSETS = os.path.join(BASE, "assets")
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)
MANIFEST = os.path.join(DATA_DIR, "articles.json")

TPL_DIR = os.path.join(BASE, "templates")
ARTICLE_TPL = os.path.join(TPL_DIR, "article.html.j2")
ARCHIVE_TPL = os.path.join(TPL_DIR, "articles_index.html.j2")
INDEX = os.path.join(BASE, "index.html")

INTERESTS = [
    "911 theories",
    "false flags",
    "aliens",
    "comet 3i atlas",
    "ufos",
    "joe rogan podcasts",
    "president trump"
]

NEWS_LOOKBACK_HOURS = 24
NUM_PER_RUN = 4
SLIDER_LATEST = 4
HEADLINES_COUNT = 10
TRENDING_COUNT = 8
TICKER_COUNT = 12
FALLBACK = "assets/fallback-hero.jpg"

# ===== Helpers =====
def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "article"

def fingerprint(title: str, url: str) -> str:
    base = (title or "").strip().lower() + "|" + (url or "").strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def load_manifest() -> List[Dict]:
    # migrate from old root file if present
    legacy = os.path.join(BASE, "articles.json")
    if (not os.path.exists(MANIFEST)) and os.path.exists(legacy):
        with open(legacy, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                save_manifest(data)
            except Exception:
                pass
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            return []
    # backfill missing fingerprints
    changed = False
    for a in data:
        if "fingerprint" not in a:
            url = a.get("source_url") or a.get("href") or a.get("file") or ""
            a["fingerprint"] = fingerprint(a.get("title",""), url)
            changed = True
    if changed:
        save_manifest(data)
    return data

def save_manifest(items: List[Dict]):
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

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
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub  = (item.findtext("pubDate") or "").strip()
            dt = parsedate_to_datetime(pub) if pub else datetime.utcnow()
            items.append({"title": title, "link": link, "date": dt})
        return items
    except Exception as e:
        print("RSS error:", e)
        return []

def openai_article(topic: str, source_title: str, source_url: str) -> str:
    # ~800 words, straight-news, skeptical crime-desk edge
    prompt = (
        "Write ~800 words in straight newspaper style (inverted pyramid). "
        "Neutral tone; careful attributions; clear sourcing. Use HTML <p>, optional <h2>, <blockquote> only. "
        "Let subtle skepticism show via sourcing and detail selection, like a meticulous crime-desk reporter.\n\n"
        f"Topic: {topic}\nHeadline: {source_title}\nSource URL: {source_url}\n"
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

def download_image(url: str, filename_hint: str) -> str:
    try:
        os.makedirs(ASSETS, exist_ok=True)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        ext = ".jpg"
        if "image/png" in resp.headers.get("Content-Type",""):
            ext = ".png"
        fname = f"{filename_hint}{ext}"
        path = os.path.join(ASSETS, fname)
        with open(path, "wb") as f:
            f.write(resp.content)
        return f"assets/{fname}"
    except Exception as e:
        print("Image download error:", e)
        return FALLBACK

def unsplash_unique(query: str, used_ids: set) -> str:
    if not UNSPLASH_KEY:
        return FALLBACK
    for _ in range(5):
        try:
            r = requests.get(
                "https://api.unsplash.com/photos/random",
                headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"},
                params={"query": query, "orientation": "landscape", "content_filter": "high"},
                timeout=20
            )
            r.raise_for_status()
            data = r.json()
            img_id = data.get("id")
            img_url = data.get("urls", {}).get("regular")
            if not img_id or not img_url:  # try again
                continue
            if img_id in used_ids:         # avoid duplicates in the same batch
                continue
            used_ids.add(img_id)
            return download_image(img_url, slugify(f"{query}-{img_id}"))
        except Exception as e:
            print("Unsplash error:", e)
            break
    return FALLBACK

def replace_block(html_text: str, start: str, end: str, inner: str) -> str:
    block = f"{start}\n{inner}\n{end}"
    return re.sub(re.escape(start) + r".*?" + re.escape(end), block, html_text, flags=re.S)

def render_article_page(title: str, meta_description: str, hero_repo_rel: str, body_html: str, ts: str) -> str:
    with open(ARTICLE_TPL, "r", encoding="utf-8") as f:
        tpl = Template(f.read())
    return tpl.render(title=title, meta_description=meta_description, hero="../"+hero_repo_rel, body=body_html, timestamp=ts)

def render_archive(manifest: List[Dict]):
    with open(ARCHIVE_TPL, "r", encoding="utf-8") as f:
        tpl = Template(f.read())
    articles = [{
        "href": f"./{m['file']}",
        "title": m["title"],
        "image": m.get("image", FALLBACK),
        "date": m["date"]
    } for m in manifest]
    html_out = tpl.render(articles=articles)
    os.makedirs(ART_DIR, exist_ok=True)
    with open(os.path.join(ART_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_out)

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

# ===== Main =====
def main():
    os.makedirs(ART_DIR, exist_ok=True)
    manifest = load_manifest()
    seen = {a["fingerprint"] for a in manifest if a.get("fingerprint")}
    cutoff = datetime.utcnow() - timedelta(hours=NEWS_LOOKB_
