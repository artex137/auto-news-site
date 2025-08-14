import os
import re
import json
import hashlib
import html
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Set
from urllib.parse import urlparse
import difflib
import requests
try:
    from bs4 import BeautifulSoup  # optional
except Exception:
    BeautifulSoup = None
from jinja2 import Template

# ===== Config =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_KEY   = os.getenv("UNSPLASH_KEY")

BASE     = os.path.dirname(os.path.abspath(__file__))
ART_DIR  = os.path.join(BASE, "articles")
ASSETS   = os.path.join(BASE, "assets")
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)
MANIFEST = os.path.join(DATA_DIR, "articles.json")
INTERESTS_FILE = os.path.join(DATA_DIR, "interests.txt")

TPL_DIR       = os.path.join(BASE, "templates")
ARTICLE_TPL   = os.path.join(TPL_DIR, "article.html.j2")
ARCHIVE_TPL   = os.path.join(TPL_DIR, "articles_index.html.j2")
INDEX         = os.path.join(BASE, "index.html")

# Fallback interests if interests.txt is missing
DEFAULT_INTERESTS = [
    "911 theories",
    "false flags",
    "aliens",
    "comet 3i atlas",
    "ufos",
    "joe rogan podcasts",
    "president trump",
]

NEWS_LOOKBACK_HOURS = 24
DEDUP_WINDOW_HOURS  = 72   # cross-run same-story suppression window
NUM_PER_RUN    = 4
SLIDER_LATEST  = 4
FEATURED_COUNT = 6
HEADLINES_COUNT= 10
TRENDING_COUNT = 8
TICKER_COUNT   = 12
FALLBACK       = "assets/fallback-hero.jpg"

# Exclude ads/retail
RETAIL_DOMAINS = {
    "amazon.com","amazon.ca","ebay.com","bestbuy.com","bestbuy.ca","walmart.com","walmart.ca",
    "newegg.com","newegg.ca","aliexpress.com","dell.com","alienware.com","lenovo.com","hp.com",
    "bhphotovideo.com","microcenter.com","store.google.com","store.apple.com","apple.com",
    "samsung.com","asus.com","msi.com"
}
AD_KEYWORDS = {"sale","deal","discount","coupon","buy","order now","% off","save","preorder","pre-order","promo","promo code","sponsored","advertorial","ad:"}

STOPWORDS = {
    "the","a","an","and","or","but","for","nor","on","in","at","to","from","by","of","with",
    "as","is","are","was","were","be","been","being","that","this","those","these","it","its",
    "into","about","after","before","over","under","near","new","latest","breaking","update",
    "report","reports","says","say","said","reveals","revealed","revealing"
}

# ===== Time =====
def as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def parse_manifest_dt(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M %Z").replace(tzinfo=timezone.utc)
    except Exception:
        return None

# ===== Helpers =====
def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "article"

def fingerprint_key(title: str, url: str) -> str:
    base = (title or "").strip().lower() + "|" + (url or "").strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def load_interests() -> List[str]:
    if os.path.exists(INTERESTS_FILE):
        try:
            with open(INTERESTS_FILE, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
            return [ln for ln in lines if ln]
        except Exception:
            pass
    return DEFAULT_INTERESTS[:]

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
    changed = False
    for a in data:
        if "fingerprint" not in a:
            url = a.get("source_url") or a.get("href") or a.get("file") or ""
            a["fingerprint"] = fingerprint_key(a.get("title",""), url); changed = True
    if changed:
        save_manifest(data)
    return data

def save_manifest(items: List[Dict]):
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

def is_advertorial(title: str, link: str) -> bool:
    t = (title or "").lower()
    if any(k in t for k in AD_KEYWORDS):
        return True
    try:
        host = urlparse(link).netloc.lower().split(":")[0]
    except Exception:
        host = ""
    if host.startswith("www."): host = host[4:]
    if any(d in host for d in RETAIL_DOMAINS):
        return True
    if any(seg in link.lower() for seg in ["/shop","/shopping","/deals","/deal/","/store/","/buy/"]):
        return True
    return False

def google_news_rss(query: str, limit: int = 20) -> List[Dict]:
    # Strict: only search for each interest, with retailer exclusions
    exclusions = " ".join([f"-site:{d}" for d in ["amazon.com","dell.com","alienware.com","bestbuy.com","newegg.com","walmart.com","ebay.com"]])
    q = urllib.parse.quote(f"{query} {exclusions}")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, timeout=15); r.raise_for_status()
        root = ET.fromstring(r.text)
        items = []
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()
            if not title or not link: continue
            if is_advertorial(title, link): continue
            dt = as_utc(parsedate_to_datetime(pub)) if pub else utcnow()
            items.append({"title": title, "link": link, "date": dt})
        return items
    except Exception as e:
        print("RSS error:", e); return []

def fetch_page_text(url: str, max_chars: int = 7000) -> str:
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"}); r.raise_for_status()
        text = r.text
        if not BeautifulSoup:
            stripped = re.sub("<script.*?</script>", " ", text, flags=re.S|re.I)
            stripped = re.sub("<style.*?</style>",  " ", stripped, flags=re.S|re.I)
            stripped = re.sub("<[^>]+>", " ", stripped)
            return re.sub(r"\s+", " ", stripped).strip()[:max_chars]
        soup = BeautifulSoup(text, "lxml")
        article = soup.find("article")
        if article:
            content = " ".join(p.get_text(" ", strip=True) for p in article.find_all(["p","li"]))
        else:
            candidates = []
            for sel in ["main","div","section"]:
                for el in soup.find_all(sel):
                    cls = " ".join(el.get("class", [])); idv = el.get("id","")
                    score = 0
                    if any(k in cls.lower() for k in ["content","article","story","post","entry","body"]): score += 2
                    if any(k in idv.lower() for k in ["content","article","story","post","entry","body"]): score += 2
                    score += min(len(el.find_all("p")), 6)
                    if score >= 4: candidates.append((score, el))
            if candidates:
                best = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
                content = " ".join(p.get_text(" ", strip=True) for p in best.find_all("p"))
            else:
                content = soup.get_text(" ", strip=True)
        content = re.sub(r"\s+", " ", content).strip()
        return content[:max_chars]
    except Exception as e:
        print("parse error:", e); return ""

# ===== Title similarity across runs =====
def tokenize_title(title: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (title or "").lower())
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit()]

def topic_signature(title: str) -> Set[str]:
    toks = tokenize_title(title)
    seen, ordered = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); ordered.append(t)
        if len(ordered) >= 8: break
    return set(ordered)

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

def is_same_story(title: str, recent_titles: List[str], recent_sigs: List[Set[str]]) -> bool:
    sig = topic_signature(title)
    for rt, rs in zip(recent_titles, recent_sigs):
        ratio = difflib.SequenceMatcher(None, title.lower(), rt.lower()).ratio()
        if ratio >= 0.82: return True
        if jaccard(sig, rs) >= 0.6: return True
        if len(sig) >= 4 and len(sig & rs) >= len(sig) - 1: return True
    return False

def openai_article(topic: str, source_title: str, source_url: str, source_text: str) -> str:
    if not OPENAI_API_KEY: raise RuntimeError("OPENAI_API_KEY missing")
    context = source_text[:7000] if source_text else ""
    prompt = (
        "Write ~800 words in an outrageous, sensational, skeptical newspaper tone (fast, punchy, provocative) while being truthful to the provided source text.\n"
        "MANDATORY: Include specific facts (names, dates, locations, agencies, charges, direct quotes, numbers, timelines). No vague summary.\n"
        "Attribute claims (e.g., 'according to ...'). Use only <p>, <h2>, <blockquote> HTML.\n\n"
        f"HEADLINE: {source_title}\nTOPIC: {topic}\nSOURCE URL: {source_url}\n"
        "SOURCE TEXT (verbatim; rely on this for facts):\n"
        f"{context}\n"
    )
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role":"system","content":"You are a sensational, skeptical columnist who adheres strictly to the provided source text; avoid fabrications or hate."},
                {"role":"user","content": prompt},
            ],
            "temperature": 0.6,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def download_image(url: str, filename_hint: str) -> str:
    try:
        os.makedirs(ASSETS, exist_ok=True)
        resp = requests.get(url, timeout=30); resp.raise_for_status()
        ext = ".jpg"; ctype = resp.headers.get("Content-Type","")
        if "image/png" in ctype: ext = ".png"
        fname = f"{filename_hint}{ext}"
        with open(os.path.join(ASSETS, fname), "wb") as f: f.write(resp.content)
        return f"assets/{fname}"
    except Exception as e:
        print("Image download error:", e); return FALLBACK

def unsplash_unique(query: str, used_ids: Set[str]) -> str:
    if not UNSPLASH_KEY: return FALLBACK
    for _ in range(5):
        try:
            r = requests.get(
                "https://api.unsplash.com/photos/random",
                headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"},
                params={"query": query, "orientation": "landscape", "content_filter":"high"},
                timeout=20,
            ); r.raise_for_status()
            data   = r.json()
            img_id = data.get("id")
            img_url = data.get("urls",{}).get("regular")
            if not img_id or not img_url: continue
            if img_id in used_ids: continue
            used_ids.add(img_id)
            return download_image(img_url, slugify(f"{query}-{img_id}"))
        except Exception as e:
            print("Unsplash error:", e); break
    return FALLBACK

def replace_block(html_text: str, start: str, end: str, inner: str) -> str:
    block = f"{start}\n{inner}\n{end}"
    return re.sub(re.escape(start)+r".*?"+re.escape(end), block, html_text, flags=re.S)

def render_article_page(title: str, meta_description: str, hero_repo_rel: str, body_html: str, ts: str, sources_pretty: List[Dict]) -> str:
    with open(ARTICLE_TPL, "r", encoding="utf-8") as f:
        tpl = Template(f.read())
    return tpl.render(
        title=title,
        meta_description=meta_description,
        hero="../"+hero_repo_rel,
        body=body_html,
        timestamp=ts,
        sources=sources_pretty
    )

def render_archive(manifest: List[Dict]):
    with open(ARCHIVE_TPL, "r", encoding="utf-8") as f:
        tpl = Template(f.read())
    articles = [{"href": f"./{m['file']}", "title": m["title"], "image": m.get("image", FALLBACK), "date": m["date"]} for m in manifest]
    html_out = tpl.render(articles=articles)
    os.makedirs(ART_DIR, exist_ok=True)
    with open(os.path.join(ART_DIR, "index.html"), "w", encoding="utf-8") as f: f.write(html_out)

def toronto_weather() -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude":43.65107,"longitude":-79.347015,"current":"temperature_2m,apparent_temperature,wind_speed_10m","timezone":"America/Toronto"}
    try:
        r = requests.get(url, params=params, timeout=15); r.raise_for_status()
        cur = r.json().get("current", {})
        t = cur.get("temperature_2m"); feels = cur.get("apparent_temperature"); wind = cur.get("wind_speed_10m")
        if t is None: return "Weather unavailable"
        return f"{t}°C (feels {feels}°C), wind {wind} km/h"
    except Exception:
        return "Weather unavailable"

def pick_diverse(candidates: List[Dict], k: int) -> List[Dict]:
    chosen, seen_topic, seen_domain = [], set(), set()
    for c in sorted(candidates, key=lambda x: x["date"], reverse=True):
        dom = urlparse(c["link"]).netloc.lower().split(":")[0]
        if dom.startswith("www."): dom = dom[4:]
        if (c["topic"] in seen_topic) or (dom in seen_domain): continue
        chosen.append(c); seen_topic.add(c["topic"]); seen_domain.add(dom)
        if len(chosen) >= k: return chosen
    for c in sorted(candidates, key=lambda x: x["date"], reverse=True):
        if c in chosen: continue
        chosen.append(c)
        if len(chosen) >= k: break
    return chosen[:k]

# ===== Main =====
def main():
    os.makedirs(ART_DIR, exist_ok=True)
    interests = load_interests()
    manifest  = load_manifest()
    seen_fps  = {a["fingerprint"] for a in manifest if a.get("fingerprint")}
    cutoff    = utcnow() - timedelta(hours=NEWS_LOOKBACK_HOURS)
    dedup_cut = utcnow() - timedelta(hours=DEDUP_WINDOW_HOURS)

    # recent memory for cross-run same-story suppression
    recent_titles: List[str] = []
    recent_sigs:   List[Set[str]] = []
    for a in manifest:
        dt = parse_manifest_dt(a.get("date",""))
        if dt and dt >= dedup_cut:
            ttl = a.get("title","")
            recent_titles.append(ttl)
            recent_sigs.append(topic_signature(ttl))

    # strictly query each interest
    candidates: List[Dict] = []
    for topic in interests:
        for it in google_news_rss(topic, limit=12):
            it_date = as_utc(it["date"])
            if it_date < cutoff: continue
            fp = fingerprint_key(it["title"], it["link"])
            if fp in seen_fps: continue
            if is_same_story(it["title"], recent_titles, recent_sigs): continue
            candidates.append({"topic": topic, "title": it["title"], "link": it["link"], "date": it_date})

    uniq = pick_diverse(candidates, NUM_PER_RUN)

    # Generate articles
    used_img_ids: Set[str] = set()
    created: List[Dict] = []

    for c in uniq:
        source_text = fetch_page_text(c["link"])
        try:
            body_html = openai_article(c["topic"], c["title"], c["link"], source_text)
        except Exception as e:
            print("OpenAI error:", e); continue

        # Pretty sources (no raw full URL text)
        def nice_label(u: str) -> str:
            try:
                h = urlparse(u).netloc.lower()
                if h.startswith("www."): h = h[4:]
                return h
            except Exception:
                return "Source"
        sources_pretty = [{"url": c["link"], "label": nice_label(c["link"])}]

        img_repo_rel = unsplash_unique(c["topic"], used_img_ids)
        filename = f"{slugify(c['title'])}.html"
        ts = c["date"].strftime("%Y-%m-%d %H:%M UTC")

        page_html = render_article_page(c["title"], c["title"], img_repo_rel, body_html, ts, sources_pretty)
        with open(os.path.join(ART_DIR, filename), "w", encoding="utf-8") as f: f.write(page_html)

        fp = fingerprint_key(c["title"], c["link"])
        manifest.insert(0, {"title": c["title"], "file": filename, "image": img_repo_rel, "date": ts, "source_url": c["link"], "fingerprint": fp})
        created.append(c)

        # update memory in-run
        recent_titles.append(c["title"]); recent_sigs.append(topic_signature(c["title"]))

    if created:
        save_manifest(manifest)
        render_archive(manifest)

    # Homepage blocks
    latest = manifest[:SLIDER_LATEST]
    slides_html = "\n".join([
        ("<a class='slide' href='articles/{file}' style=\"background-image:url('{img}')\">"
         "<div class='slide-content'><div class='kicker'>Top Story</div>"
         "<h2 class='slide-headline'>{ttl}</h2></div></a>").format(
            file=m["file"], img=m.get("image", FALLBACK), ttl=html.escape(m["title"])
        ) for m in latest
    ])

    featured_src = manifest[SLIDER_LATEST:SLIDER_LATEST+FEATURED_COUNT]
    featured_html = "\n".join([
        ("<a class='card' href='articles/{file}'>"
         "<div class='card-media' style=\"background-image:url('{img}')\"></div>"
         "<div class='card-body'><h4 class='card-title'>{ttl}</h4></div>"
         "</a>").format(file=m["file"], img=m.get("image", FALLBACK), ttl=html.escape(m["title"]))
        for m in featured_src
    ])

    headlines_src = manifest[:HEADLINES_COUNT]
    headlines_html = "\n".join([f"<li><a href=\"articles/{m['file']}\">{html.escape(m['title'])}</a></li>" for m in headlines_src])

    # ticker built from interests (strict)
    ticker_titles: List[str] = []
    for topic in interests:
        for itm in google_news_rss(topic, limit=2):
            ticker_titles.append(itm["title"])
            if len(ticker_titles) >= TICKER_COUNT: break
        if len(ticker_titles) >= TICKER_COUNT: break
    ticker_text = " · ".join(ticker_titles) if ticker_titles else "Fresh updates every cycle."

    # trending can remain general
    try:
        url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
        r = requests.get(url, timeout=15); r.raise_for_status()
        root = ET.fromstring(r.text)
        trending = []
        for item in root.findall(".//item")[:TRENDING_COUNT]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            if not title or not link: continue
            if is_advertorial(title, link): continue
            trending.append({"title": title, "link": link})
        trending_html = "\n".join([f"<li><a href=\"{i['link']}\" target=\"_blank\" rel=\"noopener\">{html.escape(i['title'])}</a></li>" for i in trending]) or "<li>No data.</li>"
    except Exception:
        trending_html = "<li>No data.</li>"

    weather_text = toronto_weather()

    with open(INDEX, "r", encoding="utf-8") as f: idx = f.read()
    idx = replace_block(idx, "<!--SLIDES-->",   "<!--/SLIDES-->",   slides_html)
    idx = replace_block(idx, "<!--FEATURED-->", "<!--/FEATURED-->", featured_html)
    idx = replace_block(idx, "<!--HEADLINES-->", "<!--/HEADLINES-->", headlines_html)
    idx = replace_block(idx, "<!--TICKER-->",   "<!--/TICKER-->",   ticker_text)
    idx = replace_block(idx, "<!--TRENDING-->", "<!--/TRENDING-->", trending_html)
    idx = replace_block(idx, "<!--WEATHER-->",  "<!--/WEATHER-->",  weather_text)
    with open(INDEX, "w", encoding="utf-8") as f: f.write(idx)

if __name__ == "__main__":
    main()
