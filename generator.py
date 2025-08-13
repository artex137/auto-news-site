import os
import re
import json
import hashlib
import html
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
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

# Expanded interest list (conspiracy + adjacent topics; mainstream sources via Google News)
INTERESTS = [
    "UAP disclosure", "UFO hearings", "AARO", "NASA UAP panel", "Area 51", "Roswell incident",
    "Phoenix Lights", "Tic Tac UFO", "Skinwalker Ranch",
    "Project Blue Beam", "MKUltra", "COINTELPRO", "Operation Northwoods", "Operation Mockingbird",
    "black budget programs", "DARPA projects", "surveillance state", "mass data collection",
    "facial recognition policy", "whistleblower", "deep state", "Bilderberg", "Skull and Bones",
    "WEF criticism", "Epstein case", "Ghislaine Maxwell", "Assange extradition",
    "geoengineering", "HAARP", "weather modification", "FEMA emergency planning",
    "digital ID", "CBDC policy", "censorship and moderation", "misinformation policy",
    "false flag debate", "military industrial complex",
]

NEWS_LOOKBACK_HOURS = 24
NUM_PER_RUN = 4
SLIDER_LATEST = 4
FEATURED_COUNT = 6
HEADLINES_COUNT = 10
TRENDING_COUNT = 8
TICKER_COUNT = 12
FALLBACK = "assets/fallback-hero.jpg"

# ===== Time helpers (UTC-aware) =====
def as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

# ===== Helpers =====
def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "article"

def fingerprint(title: str, url: str) -> str:
    base = (title or "").strip().lower() + "|" + (url or "").strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def load_manifest() -> List[Dict]:
    legacy = os.path.join(BASE, "articles.json")
    if (not os.path.exists(MANIFEST)) and os.path.exists(legacy):
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                data = json.load(f)
            save_manifest(data)
        except Exception:
            pass
    if not os.path.exists(MANIFEST):
        return []
    try:
        with open(MANIFEST, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    # Backfill missing fingerprints
    changed = False
    for a in data:
        if "fingerprint" not in a:
            url = a.get("source_url") or a.get("href") or a.get("file") or ""
            a["fingerprint"] = fingerprint(a.get("title", ""), url)
            changed = True
    if changed:
        save_manifest(data)
    return data

def save_manifest(items: List[Dict]):
    with open(MANIFEST, "w", encoding="utf-8") as f:
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
            pub = (item.findtext("pubDate") or "").strip()
            dt = as_utc(parsedate_to_datetime(pub)) if pub else utcnow()
            items.append({"title": title, "link": link, "date": dt})
        return items
    except Exception as e:
        print("RSS error:", e)
        return []

def openai_article(topic: str, source_title: str, source_url: str) -> str:
    # Sensational, rage-baiting voice — but still anchored to sourced info and without fabricating facts or targeting protected groups.
    prompt = (
        "Write approximately 800 words as a wildly outrageous, razor-sharp, rage-baiting sensationalist who never trusts the official line. "
        "Voice: punchy, provocative, relentless skepticism, clever metaphors, rapid-fire transitions. "
        "BUT: do not invent facts. Base all claims on the source; hedge with attributions (\"according to...\"). "
        "No slurs or attacks on protected classes. Use only <p>, <h2>, <blockquote> HTML tags.\n\n"
        f"Topic: {topic}\nHeadline: {source_title}\nSource URL: {source_url}\n"
    )
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a sensational, skeptical columnist who still adheres to verifiable sourcing and avoids hate."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
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
        ctype = resp.headers.get("Content-Type", "")
        if "image/png" in ctype:
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
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            img_id = data.get("id")
            img_url = data.get("urls", {}).get("regular")
            if not img_id or not img_url:
                continue
            if img_id in used_ids:
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
    return tpl.render(
        title=title,
        meta_description=meta_description,
        hero="../" + hero_repo_rel,
        body=body_html,
        timestamp=ts,
    )

def render_archive(manifest: List[Dict]):
    with open(ARCHIVE_TPL, "r", encoding="utf-8") as f:
        tpl = Template(f.read())
    articles = [
        {"href": f"./{m['file']}", "title": m["title"], "image": m.get("image", FALLBACK), "date": m["date"]}
        for m in manifest
    ]
    html_out = tpl.render(articles=articles)
    os.makedirs(ART_DIR, exist_ok=True)
    with open(os.path.join(ART_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_out)

def toronto_weather() -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 43.65107, "longitude": -79.347015,
        "current": "temperature_2m,apparent_temperature,wind_speed_10m",
        "timezone": "America/Toronto",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        cur = r.json().get("current", {})
        t = cur.get("temperature_2m"); feels = cur.get("apparent_temperature"); wind = cur.get("wind_speed_10m")
        if t is None: return "Weather unavailable"
        return f"{t}°C (feels {feels}°C), wind {wind} km/h"
    except Exception:
        return "Weather unavailable"

# ===== Main =====
def main():
    os.makedirs(ART_DIR, exist_ok=True)
    manifest = load_manifest()
    seen = {a["fingerprint"] for a in manifest if a.get("fingerprint")}
    cutoff = utcnow() - timedelta(hours=NEWS_LOOKBACK_HOURS)

    # Collect fresh candidates across topics
    candidates: List[Dict] = []
    for topic in INTERESTS:
        for it in google_news_rss(topic, limit=10):
            it_date = as_utc(it["date"])
            if it_date < cutoff:
                continue
            candidates.append({"topic": topic, "title": it["title"], "link": it["link"], "date": it_date})

    # newest first, unique by title+link
    uniq, seen_tmp = [], set()
    for c in sorted(candidates, key=lambda x: x["date"], reverse=True):
        fp = fingerprint(c["title"], c["link"])
        if fp in seen or fp in seen_tmp: continue
        uniq.append(c); seen_tmp.add(fp)
        if len(uniq) >= NUM_PER_RUN: break

    # Generate articles
    used_img_ids: set = set()
    created: List[Dict] = []
    for c in uniq:
        try:
            body_html = openai_article(c["topic"], c["title"], c["link"])
        except Exception:
            continue
        img_repo_rel = unsplash_unique(c["topic"], used_img_ids)
        filename = f"{slugify(c['title'])}.html"
        ts = c["date"].strftime("%Y-%m-%d %H:%M UTC")
        page_html = render_article_page(c["title"], c["title"], img_repo_rel, body_html, ts)
        with open(os.path.join(ART_DIR, filename), "w", encoding="utf-8") as f:
            f.write(page_html)
        fp = fingerprint(c["title"], c["link"])
        manifest.insert(0, {
            "title": c["title"], "file": filename, "image": img_repo_rel,
            "date": ts, "source_url": c["link"], "fingerprint": fp
        })
        created.append(c)

    if created:
        save_manifest(manifest)
        render_archive(manifest)

    # Build homepage blocks
    latest = manifest[:SLIDER_LATEST]
    slides_html = "\n".join(
        [
            (
                "<a class='slide' href='articles/{file}' "
                "style=\"background-image:url('{img}')\">"
                "<div class='slide-content'><div class='kicker'>Top Story</div>"
                "<h2 class='slide-headline'>{ttl}</h2></div></a>"
            ).format(file=m["file"], img=m.get("image", FALLBACK), ttl=html.escape(m["title"]))
            for m in latest
        ]
    )

    # Featured cards (next items)
    featured_src = manifest[SLIDER_LATEST:SLIDER_LATEST + FEATURED_COUNT]
    featured_html = "\n".join(
        [
            (
                "<a class='card' href='articles/{file}'>"
                "<div class='card-media' style=\"background-image:url('{img}')\"></div>"
                "<div class='card-body'><h4 class='card-title'>{ttl}</h4></div>"
                "</a>"
            ).format(file=m["file"], img=m.get("image", FALLBACK), ttl=html.escape(m["title"]))
            for m in featured_src
        ]
    )

    headlines_src = manifest[:HEADLINES_COUNT]
    headlines_html = "\n".join(
        [f"<li><a href=\"articles/{m['file']}\">{html.escape(m['title'])}</a></li>" for m in headlines_src]
    )

    # Ticker & trending
    ticker_titles: List[str] = []
    for topic in INTERESTS:
        for itm in google_news_rss(topic, limit=2):
            ticker_titles.append(itm["title"])
            if len(ticker_titles) >= TICKER_COUNT: break
        if len(ticker_titles) >= TICKER_COUNT: break
    ticker_text = " · ".join(ticker_titles) if ticker_titles else "Fresh updates every cycle."

    trending_feed = google_news_rss(None, limit=TRENDING_COUNT)
    trending_html = "\n".join(
        [f"<li><a href=\"{i['link']}\" target=\"_blank\" rel=\"noopener\">{html.escape(i['title'])}</a></li>" for i in trending_feed]
    ) or "<li>No data.</li>"

    weather_text = toronto_weather()

    # Inject into static index.html
    with open(INDEX, "r", encoding="utf-8") as f:
        idx = f.read()
    def rep(s, e, v): return replace_block(idx, s, e, v)
    idx = rep("<!--SLIDES-->", "<!--/SLIDES-->", slides_html)
    idx = rep("<!--FEATURED-->", "<!--/FEATURED-->", featured_html)
    idx = rep("<!--HEADLINES-->", "<!--/HEADLINES-->", headlines_html)
    idx = rep("<!--TICKER-->", "<!--/TICKER-->", ticker_text)
    idx = rep("<!--TRENDING-->", "<!--/TRENDING-->", trending_html)
    idx = rep("<!--WEATHER-->", "<!--/WEATHER-->", weather_text)
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(idx)

if __name__ == "__main__":
    main()
