import os
import re
import json
import hashlib
import html
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin
import difflib
import math
import requests

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None
try:
    from readability import Document
except Exception:
    Document = None
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

from jinja2 import Template
from scraper import scrape_article  # Playwright fallback

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

# Interests fallback
DEFAULT_INTERESTS = [
    "911 theories","false flags","aliens","comet 3i atlas","ufos","joe rogan podcasts","president trump",
]

# Windows / limits
NEWS_LOOKBACK_HOURS = 24
DEDUP_WINDOW_HOURS  = 72

# Per-run targets
NUM_PER_RUN    = 4
SLIDER_LATEST  = 4
FEATURED_COUNT = 6
HEADLINES_COUNT= 10
TRENDING_COUNT = 8
TICKER_COUNT   = 12
FALLBACK       = "assets/fallback-hero.jpg"

# Filters
RETAIL_DOMAINS = {
    "amazon.com","amazon.ca","ebay.com","bestbuy.com","bestbuy.ca","walmart.com","walmart.ca",
    "newegg.com","newegg.ca","aliexpress.com","dell.com","alienware.com","lenovo.com","hp.com",
    "bhphotovideo.com","microcenter.com","store.google.com","store.apple.com","apple.com",
    "samsung.com","asus.com","msi.com"
}
AD_KEYWORDS = {"sale","deal","discount","coupon","buy","order now","% off","save","preorder","pre-order","promo","promo code","sponsored","advertorial","ad:"}

STOPWORDS = {"the","a","an","and","or","but","for","nor","on","in","at","to","from","by","of","with",
             "as","is","are","was","were","be","been","being","that","this","those","these","it","its",
             "into","about","after","before","over","under","near","new","latest","breaking","update",
             "report","reports","says","say","said","reveals","revealed","revealing"}

UA_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36", "Accept-Language": "en-US,en;q=0.9"}
UA_GOOGLEBOT = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)","Accept-Language":"en-US,en;q=0.9"}

# ===== Time helpers =====
def as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def parse_manifest_dt(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M %Z").replace(tzinfo=timezone.utc)
    except Exception:
        return None

# ===== Basic helpers =====
def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "article"

def fingerprint_key(title: str, url: str) -> str:
    base = (title or "").strip().lower() + "|" + (url or "").strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def load_interests() -> List[str]:
    path = INTERESTS_FILE
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
            return [ln for ln in lines if ln]
        except Exception:
            pass
    return DEFAULT_INTERESTS[:]

def load_manifest() -> List[Dict]:
    if not os.path.exists(MANIFEST): return []
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
        with open(MANIFEST, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return data

def save_manifest(items: List[Dict]):
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

# ===== Feed & parsing =====
def is_advertorial(title: str, link: str) -> bool:
    t = (title or "").lower()
    if any(k in t for k in AD_KEYWORDS): return True
    try:
        host = urlparse(link).netloc.lower().split(":")[0]
    except Exception:
        host = ""
    if host.startswith("www."): host = host[4:]
    if any(d in host for d in RETAIL_DOMAINS): return True
    if any(seg in link.lower() for seg in ["/shop","/shopping","/deals","/deal/","/store/","/buy/"]): return True
    return False

def google_news_rss(query: str, limit: int = 20) -> List[Dict]:
    exclusions = " ".join([f"-site:{d}" for d in ["amazon.com","dell.com","alienware.com","bestbuy.com","newegg.com","walmart.com","ebay.com"]])
    q = urllib.parse.quote(f"{query} {exclusions}")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    out = []
    try:
        r = requests.get(url, timeout=15); r.raise_for_status()
        root = ET.fromstring(r.text)
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()
            if not title or not link: continue
            if is_advertorial(title, link): continue
            dt = as_utc(parsedate_to_datetime(pub)) if pub else utcnow()
            out.append({"title": title, "link": link, "date": dt})
    except Exception as e:
        print("RSS error:", e)
    return out

# ===== Extraction helpers =====
def try_jsonld_article(soup: BeautifulSoup) -> Optional[str]:
    for script in soup.find_all("script", attrs={"type":"application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        objs = data if isinstance(data, list) else [data]
        for obj in objs:
            t = obj.get("@type")
            if isinstance(t, list): t = [x.lower() for x in t]
            elif isinstance(t, str): t = [t.lower()]
            else: t = []
            if any(x in {"newsarticle","article","blogposting"} for x in t):
                body = obj.get("articleBody") or obj.get("description")
                if body and isinstance(body, str) and len(body) > 250:
                    return re.sub(r"\s+", " ", body).strip()
    return None

def extract_main_text(soup: BeautifulSoup) -> str:
    art = soup.find("article")
    if art:
        txt = " ".join(p.get_text(" ", strip=True) for p in art.find_all(["p","li"]))
        if len(txt) > 300:
            return re.sub(r"\s+", " ", txt).strip()
    candidates = []
    for sel in ["main","div","section"]:
        for el in soup.find_all(sel):
            cls = " ".join(el.get("class", [])); idv = el.get("id","")
            score = 0
            if any(k in (cls or "").lower() for k in ["content","article","story","post","entry","body","read"]): score += 4
            score += min(len(el.find_all("p")), 10)
            if score >= 8:
                candidates.append((score, el))
    if candidates:
        best = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
        txt = " ".join(p.get_text(" ", strip=True) for p in best.find_all("p"))
        return re.sub(r"\s+", " ", txt).strip()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()

def extract_with_readability(html_text: str) -> str:
    if not Document:
        return ""
    try:
        doc = Document(html_text)
        cleaned_html = doc.summary(html_partial=True)
        if BeautifulSoup:
            soup = BeautifulSoup(cleaned_html, "lxml")
            text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all(["p","li"]))
        else:
            text = re.sub("<[^>]+>", " ", cleaned_html)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""

def fetch_html(url: str, headers: Dict[str,str]) -> Tuple[str, str]:
    r = requests.get(url, timeout=20, headers=headers, allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text

def fetch_via_jina_text(url: str) -> str:
    try:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}{parsed.path or ''}"
        if parsed.query:
            base += f"?{parsed.query}"
        snapshot = f"https://r.jina.ai/{base}"
        r = requests.get(snapshot, timeout=20)
        if r.ok and len(r.text) > 250:
            return re.sub(r"\s+", " ", r.text).strip()
    except Exception:
        pass
    return ""

def youtube_video_id(url: str) -> Optional[str]:
    try:
        u = urlparse(url)
        if "youtube.com" in u.netloc:
            qs = urllib.parse.parse_qs(u.query)
            if "v" in qs: return qs["v"][0]
            parts = u.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in {"embed","shorts"}:
                return parts[1]
        if "youtu.be" in u.netloc:
            return u.path.strip("/")
    except Exception:
        pass
    return None

def fetch_youtube_transcript(url: str) -> str:
    vid = youtube_video_id(url)
    if not vid or not YouTubeTranscriptApi:
        return ""
    try:
        trs = YouTubeTranscriptApi.get_transcript(vid, languages=['en'])
        text = " ".join([t["text"] for t in trs if t.get("text")])
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""

def fetch_page_text(url: str, title: Optional[str] = None, min_chars: int = 250, max_chars: int = 7000) -> str:
    """Return article text using HTML/Readability/Jina; fallback to Playwright render."""
    # YouTube special case
    host = urlparse(url).netloc.lower()
    if "youtube.com" in host or "youtu.be" in host:
        yt = fetch_youtube_transcript(url)
        return yt[:max_chars]

    tried_urls = set()

    def _extract_from_html(u: str, html_text: str) -> str:
        if BeautifulSoup:
            soup = BeautifulSoup(html_text, "lxml")
            can = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
            if can and can.get("href") and urlparse(can["href"]).netloc:
                cu = can["href"]
                if urlparse(cu).netloc.lower() != urlparse(u).netloc.lower() and cu not in tried_urls:
                    try:
                        u2, t2 = fetch_html(cu, UA_HEADERS)
                        tried_urls.add(u2)
                        return _extract_from_html(u2, t2)
                    except Exception:
                        pass
            body = try_jsonld_article(soup)
            if not body or len(body) < min_chars:
                body = extract_main_text(soup)
            if len(body) < min_chars and Document:
                via_read = extract_with_readability(html_text)
                if len(via_read) > len(body):
                    body = via_read
            return body
        else:
            body = re.sub("<script.*?</script>", " ", html_text, flags=re.S|re.I)
            body = re.sub("<style.*?</style>",  " ", body, flags=re.S|re.I)
            body = re.sub("<[^>]+>", " ", body)
            return re.sub(r"\s+", " ", body).strip()

    try:
        final_url, html_text = fetch_html(url, UA_HEADERS)
        tried_urls.add(final_url)
        body = _extract_from_html(final_url, html_text)

        if len(body) < min_chars:
            try:
                u2, t2 = fetch_html(final_url, UA_GOOGLEBOT)
                tried_urls.add(u2)
                body2 = _extract_from_html(u2, t2)
                if len(body2) > len(body): body = body2
            except Exception:
                pass

        if len(body) < min_chars:
            via_jina = fetch_via_jina_text(final_url)
            if len(via_jina) > len(body): body = via_jina

        if len(body) < min_chars:
            print("Playwright scraping:", url)
            body_pw, _imgs_pw = scrape_article(url)
            if len(body_pw) > len(body): body = body_pw

        body = re.sub(r"\s+", " ", body or "").strip()[:max_chars]
        return body
    except Exception as e:
        print("parse error:", e)
        return ""

# ===== Similarity / scoring =====
def tokenize_title(title: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (title or "").lower())
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit() and len(t) > 2]

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
    return inter/union if union else 0.0

def is_same_story(title: str, recent_titles: List[str], recent_sigs: List[Set[str]]) -> bool:
    sig = topic_signature(title)
    for rt, rs in zip(recent_titles, recent_sigs):
        ratio = difflib.SequenceMatcher(None, title.lower(), rt.lower()).ratio()
        if ratio >= 0.82: return True
        if jaccard(sig, rs) >= 0.6: return True
        if len(sig) >= 4 and len(sig & rs) >= len(sig) - 1: return True
    return False

def recency_score(dt: datetime) -> float:
    hours = max(0.0, (utcnow() - dt).total_seconds() / 3600.0)
    return math.exp(-hours / 24.0)

def novelty_score(title: str, recent_titles: List[str], recent_sigs: List[Set[str]]) -> float:
    sig = topic_signature(title)
    best = 0.0
    for rt, rs in zip(recent_titles, recent_sigs):
        ratio = difflib.SequenceMatcher(None, title.lower(), rt.lower()).ratio()
        jac   = jaccard(sig, rs)
        sim   = max(ratio, jac)
        if sim > best: best = sim
    return max(0.0, 1.0 - best)

def score_candidate(title: str, dt: datetime, domain: str, recent_titles: List[str], recent_sigs: List[Set[str]], recent_domains: Set[str], token_freq: Dict[str,int]) -> float:
    r = recency_score(dt)
    n = novelty_score(title, recent_titles, recent_sigs)
    dom_penalty = 0.12 if domain in recent_domains else 0.0
    sig = topic_signature(title)
    over = max((token_freq.get(t,0) for t in sig), default=0)
    tok_penalty = min(0.20, 0.04 * over)
    return 0.55*r + 0.35*n - dom_penalty - tok_penalty

def build_recent_memory(manifest: List[Dict], window_hours: int) -> Tuple[List[str], List[Set[str]], Set[str], Dict[str,int]]:
    cut = utcnow() - timedelta(hours=window_hours)
    titles: List[str] = []
    sigs:   List[Set[str]] = []
    domains: Set[str] = set()
    token_freq: Dict[str,int] = {}
    for a in manifest:
        dt = parse_manifest_dt(a.get("date",""))
        if dt and dt >= cut:
            title = a.get("title","")
            titles.append(title)
            ts = topic_signature(title)
            sigs.append(ts)
            for t in ts:
                token_freq[t] = token_freq.get(t, 0) + 1
            link = a.get("source_url","")
            d = urlparse(link).netloc.lower().split(":")[0]
            if d.startswith("www."): d = d[4:]
            if d: domains.add(d)
    return titles, sigs, domains, token_freq

# ===== Writer =====
def openai_article(topic: str, source_title: str, source_urls: List[str], source_text: str) -> str:
    if not OPENAI_API_KEY: raise RuntimeError("OPENAI_API_KEY missing")
    context = source_text[:7000] if source_text else ""
    src_list = "\n".join(f"- {u}" for u in source_urls[:6])
    prompt = (
        "Write ~800 words in an outrageous, sensational, skeptical newspaper tone while being faithful to the SOURCE TEXT.\n"
        "MANDATORY: include concrete facts (names, dates, locations, agencies, charges, quotes, numbers, timelines). Attribute claims. "
        "Output HTML using only <p>, <h2>, <blockquote>.\n\n"
        f"HEADLINE: {source_title}\nTOPIC: {topic}\nSOURCES:\n{src_list}\n\nSOURCE TEXT:\n{context}\n"
    )
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role":"system","content":"You are a sensational but accurate columnist. Never fabricate facts; stick to the provided source text."},
                {"role":"user","content": prompt},
            ],
            "temperature": 0.6,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ===== Images (Unsplash only) =====
def download_image(url: str, filename_hint: str) -> str:
    try:
        os.makedirs(ASSETS, exist_ok=True)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        ext = ".jpg"
        ctype = resp.headers.get("Content-Type","")
        if "image/png" in ctype: ext = ".png"
        fname = f"{filename_hint}{ext}"
        with open(os.path.join(ASSETS, fname), "wb") as f:
            f.write(resp.content)
        return f"assets/{fname}"
    except Exception as e:
        print("Image download error:", e)
        return FALLBACK

def unsplash_unique(query: str, used_ids: Set[str]) -> str:
    if not UNSPLASH_KEY: return FALLBACK
    for _ in range(5):
        try:
            r = requests.get(
                "https://api.unsplash.com/photos/random",
                headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"},
                params={"query": query, "orientation": "landscape", "content_filter":"high"},
                timeout=20,
            )
            r.raise_for_status()
            data   = r.json()
            img_id = data.get("id")
            img_url = data.get("urls",{}).get("regular")
            if not img_id or not img_url: continue
            if img_id in used_ids: continue
            used_ids.add(img_id)
            return download_image(img_url, slugify(f"{query}-{img_id}"))
        except Exception as e:
            print("Unsplash error:", e)
            break
    return FALLBACK

# ===== HTML block replace =====
def replace_block(html_text: str, start: str, end: str, inner: str) -> str:
    block = f"{start}\n{inner}\n{end}"
    return re.sub(re.escape(start)+r".*?"+re.escape(end), block, html_text, flags=re.S)

# ===== Renderers =====
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
    with open(os.path.join(ART_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_out)

# ===== Weather (Toronto) =====
def toronto_weather() -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude":43.65107,"longitude":-79.347015,"current":"temperature_2m,apparent_temperature,wind_speed_10m","timezone":"America/Toronto"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        cur = r.json().get("current", {})
        t = cur.get("temperature_2m"); feels = cur.get("apparent_temperature"); wind = cur.get("wind_speed_10m")
        if t is None: return "Weather unavailable"
        return f"{t}°C (feels {feels}°C), wind {wind} km/h"
    except Exception:
        return "Weather unavailable"

# ===== Selection helpers =====
def build_recent_memory(manifest: List[Dict], window_hours: int) -> Tuple[List[str], List[Set[str]], Set[str], Dict[str,int]]:
    cut = utcnow() - timedelta(hours=window_hours)
    titles: List[str] = []
    sigs:   List[Set[str]] = []
    domains: Set[str] = set()
    token_freq: Dict[str,int] = {}
    for a in manifest:
        dt = parse_manifest_dt(a.get("date",""))
        if dt and dt >= cut:
            title = a.get("title","")
            titles.append(title)
            ts = topic_signature(title)
            sigs.append(ts)
            for t in ts:
                token_freq[t] = token_freq.get(t, 0) + 1
            link = a.get("source_url","")
            d = urlparse(link).netloc.lower().split(":")[0]
            if d.startswith("www."): d = d[4:]
            if d: domains.add(d)
    return titles, sigs, domains, token_freq

# ===== Main =====
def main():
    os.makedirs(ART_DIR, exist_ok=True)
    interests = load_interests()
    manifest  = load_manifest()
    seen_fps  = {a["fingerprint"] for a in manifest if a.get("fingerprint")}
    cutoff    = utcnow() - timedelta(hours=NEWS_LOOKBACK_HOURS)

    # Build novelty memory
    recent_titles, recent_sigs, recent_domains, token_freq = build_recent_memory(manifest, DEDUP_WINDOW_HOURS)

    # Gather candidates per interest
    all_candidates: List[Dict] = []
    for topic in interests:
        feed = google_news_rss(topic, limit=24)
        for it in feed:
            dt = as_utc(it["date"])
            if dt < cutoff: continue
            title = it["title"]; link = it["link"]
            fp = fingerprint_key(title, link)
            if fp in seen_fps: continue
            if is_advertorial(title, link): continue
            if is_same_story(title, recent_titles, recent_sigs): continue
            dom = urlparse(link).netloc.lower().split(":")[0]
            if dom.startswith("www."): dom = dom[4:]
            score = score_candidate(title, dt, dom, recent_titles, recent_sigs, recent_domains, token_freq)
            all_candidates.append({"topic": topic, "title": title, "link": link, "date": dt, "domain": dom, "score": score})

    # Sort within each interest
    cands_by_interest: Dict[str, List[Dict]] = {}
    for topic in interests:
        group = [c for c in all_candidates if c["topic"] == topic]
        group.sort(key=lambda x: x["score"], reverse=True)
        cands_by_interest[topic] = group

    # Iteratively select and *create* until we have NUM_PER_RUN
    used_img_ids: Set[str] = set()
    created = []
    run_tokens: Dict[str,int] = {}
    CAP_PER_TOKEN = 1
    attempts = 0
    MAX_ATTEMPTS = 50  # safety to avoid infinite loop

    def can_take(title: str) -> bool:
        sig = topic_signature(title)
        if any(run_tokens.get(t,0) >= CAP_PER_TOKEN for t in sig):
            return False
        # avoid near-duplicate within this batch
        for made in created:
            if difflib.SequenceMatcher(None, title.lower(), made["title"].lower()).ratio() >= 0.82:
                return False
        return True

    # round-robin cursor per interest
    cursors = {k:0 for k in cands_by_interest.keys()}

    while len(created) < NUM_PER_RUN and attempts < MAX_ATTEMPTS:
        attempts += 1
        progressed = False
        for topic in list(cands_by_interest.keys()):
            lst = cands_by_interest[topic]
            i = cursors[topic]
            while i < len(lst) and not can_take(lst[i]["title"]):
                i += 1
            cursors[topic] = i + 1
            if i >= len(lst):
                continue
            cand = lst[i]

            # Try to fetch rich text for this candidate
            source_text = fetch_page_text(cand["link"], title=cand["title"], min_chars=250)
            if len(source_text) < 250:
                continue  # try next candidate

            # Use Unsplash only (no scraped images) per your request
            hero_rel = unsplash_unique(cand["topic"], used_img_ids)

            # Write article
            try:
                body_html = openai_article(cand["topic"], cand["title"], [cand["link"]], source_text)
            except Exception as e:
                print("OpenAI error:", e)
                continue

            filename = f"{slugify(cand['title'])}.html"
            ts = cand["date"].strftime("%Y-%m-%d %H:%M UTC")
            sources_pretty = [{"url": cand["link"], "label": urlparse(cand["link"]).netloc.replace('www.','')}]

            page_html = render_article_page(cand["title"], cand["title"], hero_rel, body_html, ts, sources_pretty)
            with open(os.path.join(ART_DIR, filename), "w", encoding="utf-8") as f:
                f.write(page_html)

            fp = fingerprint_key(cand["title"], cand["link"])
            manifest.insert(0, {"title": cand["title"], "file": filename, "image": hero_rel, "date": ts, "source_url": cand["link"], "fingerprint": fp})
            save_manifest(manifest)  # persist incrementally

            created.append(cand)
            sig = topic_signature(cand["title"])
            for t in sig:
                run_tokens[t] = run_tokens.get(t,0) + 1
            progressed = True

            if len(created) >= NUM_PER_RUN:
                break

        if not progressed:
            # No more viable candidates across interests
            break

    # Render archive if anything new
    if created:
        render_archive(manifest)

    # ---- Homepage sections ----
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
         "<div class='card-body'><h4 class='card-title'>{ttl}</h4><div class='card-meta'>{date}</div></div>"
         "</a>").format(file=m["file"], img=m.get("image", FALLBACK), ttl=html.escape(m["title"]), date=m["date"])
        for m in featured_src
    ])

    headlines_src = manifest[:HEADLINES_COUNT]
    headlines_html = "\n".join([f"<li><a href=\"articles/{m['file']}\">{html.escape(m['title'])}</a></li>" for m in headlines_src])

    # ticker from interests
    ticker_titles: List[str] = []
    for topic in interests:
        for itm in google_news_rss(topic, limit=2):
            ticker_titles.append(itm["title"])
            if len(ticker_titles) >= TICKER_COUNT: break
        if len(ticker_titles) >= TICKER_COUNT: break
    ticker_text = " · ".join(ticker_titles) if ticker_titles else "Fresh updates every cycle."

    # trending
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

    # weather
    weather_text = toronto_weather()

    with open(INDEX, "r", encoding="utf-8") as f: idx = f.read()
    idx = replace_block(idx, "<!--SLIDES-->",   "<!--/SLIDES-->",   slides_html)
    idx = replace_block(idx, "<!--FEATURED-->", "<!--/FEATURED-->", featured_html)
    idx = replace_block(idx, "<!--HEADLINES-->", "<!--/HEADLINES-->", headlines_html)
    idx = replace_block(idx, "<!--TICKER-->",   "<!--/TICKER-->",   ticker_text)
    idx = replace_block(idx, "<!--TRENDING-->", "<!--/TRENDING-->", trending_html)
    idx = replace_block(idx, "<!--WEATHER-->",  "<!--/WEATHER-->",  weather_text)
    with open(INDEX, "w", encoding="utf-8") as f:
        f.write(idx)

if __name__ == "__main__":
    main()
