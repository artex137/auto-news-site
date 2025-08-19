# generator.py
import os, re, json, hashlib, html, urllib.parse, xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urlparse
import difflib, math, requests

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
from scraper import scrape_article

# ===== Config =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_KEY   = os.getenv("UNSPLASH_KEY")

BASE     = os.path.dirname(os.path.abspath(__file__))
ART_DIR  = os.path.join(BASE, "articles")
ASSETS   = os.path.join(BASE, "assets")
DATA_DIR = os.path.join(BASE, "data")
TPL_DIR  = os.path.join(BASE, "templates")
os.makedirs(DATA_DIR, exist_ok=True)

MANIFEST      = os.path.join(DATA_DIR, "articles.json")
INTERESTS_TXT = os.path.join(DATA_DIR, "interests.txt")
WEIGHTED_JSON = os.path.join(DATA_DIR, "interests_weighted.json")

ARTICLE_TPL   = os.path.join(TPL_DIR, "article.html.j2")
ARCHIVE_TPL   = os.path.join(TPL_DIR, "articles_index.html.j2")
INDEX         = os.path.join(BASE, "index.html")

DEFAULT_INTERESTS = [
    "UFOs","Comet 3I ATLAS"
]

# windows/limits
NEWS_LOOKBACK_HOURS      = 24
BACKFILL_LOOKBACK_HOURS  = 96
DEDUP_WINDOW_HOURS       = 72
NUM_PER_RUN              = 8   # increased
CAP_PER_TOKEN            = 2   # allow two similar topics in a run

# homepage counts
SLIDER_LATEST   = 4
FEATURED_COUNT  = 6
HEADLINES_COUNT = 10
TRENDING_COUNT  = 8
TICKER_COUNT    = 12
FALLBACK        = "assets/fallback-hero.jpg"

# ad/product filters
RETAIL_DOMAINS = {
    "amazon.com","amazon.ca","ebay.com","bestbuy.com","bestbuy.ca","walmart.com","walmart.ca",
    "newegg.com","newegg.ca","aliexpress.com","dell.com","alienware.com","lenovo.com","hp.com",
    "bhphotovideo.com","microcenter.com","store.google.com","store.apple.com","apple.com",
    "samsung.com","asus.com","msi.com"
}
AD_KEYWORDS = {"sale","deal","discount","coupon","buy","order now","% off","save","preorder","pre-order","promo","promo code","sponsored","advertorial","ad:"}
NEGATIVE_TERMS = {"alienware","laptop","gaming laptop","gaming rig","coupon","promo","deal","shopping"}

STOPWORDS = {"the","a","an","and","or","but","for","nor","on","in","at","to","from","by","of","with",
             "as","is","are","was","were","be","been","being","that","this","those","these","it","its",
             "into","about","after","before","over","under","near","new","latest","breaking","update",
             "report","reports","says","say","said","reveals","revealed","revealing"}

UA_HEADERS   = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124 Safari/537.36", "Accept-Language": "en-US,en;q=0.9"}
UA_GOOGLEBOT = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)", "Accept-Language": "en-US,en;q=0.9"}

# ===== Focused query packs (Plan A) =====
UFO_QUERIES = [
    "UFO OR UAP",
    "\"unidentified aerial phenomena\"",
    "Pentagon UAP",
    "NORAD UFO",
    "Navy UAP",
    "MUFON report",
    "UFO Canada",
    "UFO sighting police"
]

COMET_QUERIES = [
    "\"Comet 3I ATLAS\"",
    "\"3I/ATLAS\"",
    "\"C/2019 Y4\" ATLAS",
    "ATLAS comet discovery",
    "interstellar comet ATLAS"
]

REDDIT_SUBS = ["UFOs", "UAP", "HighStrangeness", "Astronomy"]

# ===== Interest weighting =====
DEFAULT_WEIGHTS = {
    "ufos": 5.0,
    "comet 3i atlas": 5.0
}

def load_interests_weighted() -> List[Tuple[str, float]]:
    # 1) interests_weighted.json (preferred)
    if os.path.exists(WEIGHTED_JSON):
        try:
            with open(WEIGHTED_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            pairs = [(k, float(v)) for k,v in data.items()]
            return [(k, max(0.5, min(5.0, w))) for k,w in pairs if k.strip()]
        except Exception:
            pass
    # 2) interests.txt -> assign default weight mapping (or 3.0 fallback)
    if os.path.exists(INTERESTS_TXT):
        try:
            with open(INTERESTS_TXT, "r", encoding="utf-8") as f:
                items = [ln.strip() for ln in f if ln.strip()]
            out = []
            for it in items:
                out.append((it, DEFAULT_WEIGHTS.get(it.lower(), 3.0)))
            return out
        except Exception:
            pass
    # 3) default list with default weights
    return [(it, DEFAULT_WEIGHTS.get(it.lower(), 3.5)) for it in DEFAULT_INTERESTS]

# ===== Helpers =====
def as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
def utcnow() -> datetime: return datetime.now(timezone.utc)
def parse_manifest_dt(s: str) -> Optional[datetime]:
    try: return datetime.strptime(s, "%Y-%m-%d %H:%M %Z").replace(tzinfo=timezone.utc)
    except Exception: return None

def slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text or "article"
def fingerprint_key(title: str, url: str) -> str:
    base = (title or "").strip().lower() + "|" + (url or "").strip().lower()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def load_manifest() -> List[Dict]:
    if not os.path.exists(MANIFEST): return []
    try:
        with open(MANIFEST, "r", encoding="utf-8") as f: data = json.load(f)
    except Exception: return []
    changed = False
    for a in data:
        if "fingerprint" not in a:
            url = a.get("source_url") or a.get("href") or a.get("file") or ""
            a["fingerprint"] = fingerprint_key(a.get("title",""), url); changed = True
    if changed:
        with open(MANIFEST, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
    return data
def save_manifest(items: List[Dict]):
    with open(MANIFEST, "w", encoding="utf-8") as f: json.dump(items, f, indent=2, ensure_ascii=False)

# ===== Interest relevance =====
INTEREST_SYNONYMS: Dict[str, Set[str]] = {
    "ufos": {"ufo","uap","alien","aliens","extraterrestrial","flying saucer","unidentified aerial"},
    "aliens": {"alien","aliens","ufo","uap","extraterrestrial"},
    "false flags": {"false flag","psyop","staged attack","provocation"},
    "911 theories": {"9/11","911","september 11","world trade center","wtc","pentagon","building 7"},
    "joe rogan podcasts": {"joe rogan","jre","rogan podcast","spotify rogan"},
    "president trump": {"donald trump","president trump","trump"},
    "comet 3i atlas": {"comet atlas","3i atlas","c/2019 y4","atlas comet","3i/atlas"}
}
def expand_interest_keywords(interest: str) -> Set[str]:
    base = {interest.lower().strip()}
    for tok in re.findall(r"[a-z0-9][a-z0-9\-]+", interest.lower()):
        if tok not in STOPWORDS and len(tok) > 2: base.add(tok)
    for key, syns in INTEREST_SYNONYMS.items():
        if key in interest.lower(): base |= syns
    if any(k in interest.lower() for k in ["ufo","uap","alien"]):
        base |= {"ufo","uap","alien","aliens","extraterrestrial"}
    return base
def passes_interest_relevance(interest: str, title: str, body: str) -> bool:
    text = (title + " " + (body or ""))[:10000].lower()
    kws  = expand_interest_keywords(interest)
    if not any(k in text for k in kws): return False
    if any(b in text for b in NEGATIVE_TERMS): return False
    return True

# ===== Feeds & parsing =====
def is_advertorial(title: str, link: str) -> bool:
    t = (title or "").lower()
    if any(k in t for k in AD_KEYWORDS): return True
    try: host = urlparse(link).netloc.lower().split(":")[0]
    except Exception: host = ""
    if host.startswith("www."): host = host[4:]
    if any(d in host for d in RETAIL_DOMAINS): return True
    if any(seg in link.lower() for seg in ["/shop","/shopping","/deals","/deal/","/store/","/buy/"]): return True
    return False

def google_news_rss(query: str, limit: int = 28) -> List[Dict]:
    q_core = f"\"{query}\"" if " " in query.strip() else query
    negatives = "-Alienware -laptop -gaming -coupon -promo -deal -shopping"
    q = urllib.parse.quote(f"{q_core} {negatives}")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    return _rss_to_items(url, limit=limit)

def bing_news_rss(query: str, limit: int = 28) -> List[Dict]:
    q = urllib.parse.quote(query)
    url = f"https://www.bing.com/news/search?q={q}&format=rss"
    return _rss_to_items(url, limit=limit)

def reddit_rss(sub: str, limit: int = 25) -> List[Dict]:
    url = f"https://www.reddit.com/r/{sub}/.rss"
    items = _rss_to_items(url, limit=limit, reddit=True)
    out = []
    for it in items:
        # Try to extract first external link from description/content
        ext = extract_external_from_reddit(it.get("description",""), it.get("link",""))
        if ext and "reddit.com" not in urlparse(ext).netloc.lower():
            it["link"] = ext
            out.append(it)
    return out

def _rss_to_items(url: str, limit: int = 28, reddit: bool = False) -> List[Dict]:
    out = []
    try:
        r = requests.get(url, timeout=15, headers=UA_HEADERS); r.raise_for_status()
        root = ET.fromstring(r.text)
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()
            if not title or not link: continue
            if is_advertorial(title, link): continue
            try:
                dt = parsedate_to_datetime(pub) if pub else datetime.utcnow()
            except Exception:
                dt = datetime.utcnow()
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            else: dt = dt.astimezone(timezone.utc)
            # Reddit items may include content:encoded
            desc = item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded") or item.findtext("description") or ""
            out.append({"title": title, "link": link, "date": dt, "description": desc})
    except Exception as e:
        print("RSS error:", e)
    return out

def extract_external_from_reddit(desc_html: str, fallback_link: str) -> Optional[str]:
    try:
        if not desc_html or not BeautifulSoup: return None
        soup = BeautifulSoup(desc_html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and "reddit.com" not in urlparse(href).netloc.lower():
                return href
    except Exception:
        pass
    return None

# ===== Extract text (HTML → text; Playwright fallback in scraper) =====
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
        if len(txt) > 300: return re.sub(r"\s+", " ", txt).strip()
    candidates = []
    for sel in ["main","div","section"]:
        for el in soup.find_all(sel):
            cls = " ".join(el.get("class", []))
            score = 0
            if any(k in (cls or "").lower() for k in ["content","article","story","post","entry","body","read"]): score += 4
            score += min(len(el.find_all("p")), 10)
            if score >= 8: candidates.append((score, el))
    if candidates:
        best = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
        txt = " ".join(p.get_text(" ", strip=True) for p in best.find_all("p"))
        return re.sub(r"\s+", " ", txt).strip()
    return ""

def fetch_html(url: str, headers: Dict[str,str]) -> Tuple[str, str]:
    r = requests.get(url, timeout=20, headers=headers, allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text

def fetch_via_jina_text(url: str) -> str:
    try:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}{parsed.path or ''}"
        if parsed.query: base += f"?{parsed.query}"
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
            if len(parts) >= 2 and parts[0] in {"embed","shorts"}: return parts[1]
        if "youtu.be" in u.netloc: return u.path.strip("/")
    except Exception:
        pass
    return None

def fetch_youtube_transcript(url: str) -> str:
    vid = youtube_video_id(url)
    if not vid or not YouTubeTranscriptApi: return ""
    try:
        trs = YouTubeTranscriptApi.get_transcript(vid, languages=['en'])
        return re.sub(r"\s+", " ", " ".join(t["text"] for t in trs if t.get("text"))).strip()
    except Exception:
        return ""

def fetch_page_text(url: str, min_chars: int = 250, max_chars: int = 7000) -> str:
    host = urlparse(url).netloc.lower()
    if "youtube.com" in host or "youtu.be" in host:
        return fetch_youtube_transcript(url)[:max_chars]
    try:
        _, html_text = fetch_html(url, UA_HEADERS)
        body = ""
        if BeautifulSoup:
            soup = BeautifulSoup(html_text, "lxml")
            body = try_jsonld_article(soup) or extract_main_text(soup)
            if len(body) < min_chars and Document:
                try:
                    doc = Document(html_text)
                    cleaned_html = doc.summary(html_partial=True)
                    soup2 = BeautifulSoup(cleaned_html, "lxml")
                    more = " ".join(p.get_text(" ", strip=True) for p in soup2.find_all(["p","li"]))
                    if len(more) > len(body): body = more
                except Exception:
                    pass
        if len(body) < min_chars:
            via_jina = fetch_via_jina_text(url)
            if len(via_jina) > len(body): body = via_jina
        if len(body) < min_chars:
            txt, _imgs = scrape_article(url)  # final fallback (also finds images)
            if len(txt) > len(body): body = txt
        return re.sub(r"\s+", " ", (body or ""))[:max_chars].strip()
    except Exception as e:
        print("parse error:", e); return ""

# ===== Similarity / scoring =====
def tokenize_title(title: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (title or "").lower())
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit() and len(t) > 2]
def topic_signature(title: str) -> Set[str]:
    out, seen = [], set()
    for t in tokenize_title(title):
        if t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= 8: break
    return set(out)
def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter/union if union else 0.0
def is_same_story(title: str, recent_titles: List[str], recent_sigs: List[Set[str]]) -> bool:
    # RELAX dedup to allow more near-duplicates through
    sig = topic_signature(title)
    for rt, rs in zip(recent_titles, recent_sigs):
        if difflib.SequenceMatcher(None, title.lower(), rt.lower()).ratio() >= 0.92: return True
        if jaccard(sig, rs) >= 0.75: return True
        if len(sig) >= 5 and len(sig & rs) >= len(sig) - 1: return True
    return False

def recency_score(dt: datetime) -> float:
    hours = max(0.0, (utcnow() - dt).total_seconds() / 3600.0)
    return math.exp(-hours / 24.0)
def novelty_score(title: str, recent_titles: List[str], recent_sigs: List[Set[str]]) -> float:
    sig = topic_signature(title); best = 0.0
    for rt, rs in zip(recent_titles, recent_sigs):
        sim = max(difflib.SequenceMatcher(None, title.lower(), rt.lower()).ratio(), jaccard(sig, rs))
        if sim > best: best = sim
    return max(0.0, 1.0 - best)
def score_candidate(title: str, dt: datetime, domain: str, recent_titles: List[str], recent_sigs: List[Set[str]], recent_domains: Set[str], token_freq: Dict[str,int], weight: float) -> float:
    """Return a score for a candidate headline.

    If the domain has appeared recently, we hard skip by returning a
    negative score. This provides stronger protection against repeating
    the same source across runs.
    """
    if domain in recent_domains:
        return -1.0

    r = recency_score(dt)
    n = novelty_score(title, recent_titles, recent_sigs)
    sig = topic_signature(title)
    over = max((token_freq.get(t,0) for t in sig), default=0)
    tok_penalty = min(0.15, 0.03 * over)
    base = 0.55*r + 0.35*n - tok_penalty
    return base * (0.75 + 0.25 * (weight/5.0))  # weight tilt

def build_recent_memory(manifest: List[Dict], window_hours: int) -> Tuple[List[str], List[Set[str]], Set[str], Dict[str,int]]:
    cut = utcnow() - timedelta(hours=window_hours)
    titles, sigs, domains, token_freq = [], [], set(), {}
    for a in manifest:
        dt = parse_manifest_dt(a.get("date",""))
        if dt and dt >= cut:
            t = a.get("title",""); titles.append(t)
            ts = topic_signature(t); sigs.append(ts)
            for tok in ts: token_freq[tok] = token_freq.get(tok, 0) + 1
            link = a.get("source_url",""); d = urlparse(link).netloc.lower().split(":")[0]
            if d.startswith("www."): d = d[4:]
            if d: domains.add(d)
    return titles, sigs, domains, token_freq

# ===== Writer (newsroom prompt kept) =====
def openai_article(topic: str, source_title: str, source_urls: List[str], source_text: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    context = source_text[:7000]
    src_list = "\n".join(f"- {u}" for u in source_urls[:6])

    SYSTEM_MSG = (
        "You are a wire-service news writer. Write in a clear, neutral, fact-first tone (inverted pyramid).\n"
        "Mimic Canadian newsroom style like CP/CTV:\n"
        "- Lead with who/what/where/when in the first sentence.\n"
        "- Keep sentences short; avoid speculation and loaded language.\n"
        "- Attribute claims (“police said”, “according to the report”).\n"
        "- Do not invent facts. If a detail is missing in SOURCE TEXT, write “not disclosed”.\n"
        "- Include precise times, locations, agencies, and named sources when present.\n"
        "- Prefer active voice, concrete verbs, plain words.\n"
        "- Use present perfect/ simple past consistently for reported events.\n\n"
        "Output clean HTML using ONLY these tags: <h2>, <p>, <ul>, <li>, <blockquote>, <strong>, <em>.\n"
        "No inline styles, no external links in body (we provide Sources separately).\n"
        "Length target: ~700–900 words if SOURCE TEXT is rich; ~300–500 words if sparse."
    )

    USER_TMPL = (
        "Write a straight-news article that mirrors the tone and structure of the sample.\n\n"
        f"HEADLINE: {source_title}\n"
        f"TOPIC: {topic}\n\n"
        "STRUCTURE:\n"
        "1) Lede (1 paragraph): who/what/where/when; injuries/fatalities if any.\n"
        "2) Nutgraf (1 paragraph): why this matters / what authorities are doing now.\n"
        "3) Key facts (bulleted): times, locations, numbers, agencies, arrests, status.\n"
        "4) Timeline (subhead + 1–3 short paragraphs): what happened in order.\n"
        "5) Official statements (quotes, attributed in <blockquote>).\n"
        "6) Context (subhead + 1–2 short paragraphs): prior incidents, policies, trends.\n"
        "7) What’s next (1 short paragraph): investigations, charges, deadlines, next steps.\n\n"
        "CONSTRAINTS:\n"
        "- Use only details from SOURCE TEXT below. Do NOT speculate or add outside facts.\n"
        "- If a required detail is missing, write “not disclosed”.\n"
        "- Use metric units and local spellings when present in SOURCE TEXT.\n"
        "- Keep proper nouns exactly as in SOURCE TEXT.\n\n"
        "RENDERING RULES:\n"
        "- Use <h2> to label sections “Key facts”, “Timeline”, “Context”, “What’s next”.\n"
        "- Use <ul><li> for Key facts (3–8 bullets max).\n"
        "- Quotes must be inside <blockquote> with attribution in the same block.\n"
        "- Use <strong> sparingly for crucial numbers/names once.\n\n"
        "SOURCES (for reference list, not to cite inline):\n"
        f"{src_list}\n\n"
        "SOURCE TEXT:\n"
        f"{context}"
    )

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": USER_TMPL},
            ],
            "temperature": 0.6,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ===== Images =====
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
    for _ in range(6):
        try:
            r = requests.get(
                "https://api.unsplash.com/photos/random",
                headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"},
                params={"query": query, "orientation": "landscape", "content_filter":"high"},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            img_id = data.get("id"); img_url = data.get("urls",{}).get("regular")
            if not img_id or not img_url: continue
            if img_id in used_ids: continue
            used_ids.add(img_id)
            return download_image(img_url, slugify(f"{query}-{img_id}")[:50])
        except Exception as e:
            print("Unsplash error:", e); break
    return FALLBACK

# ===== HTML block replace =====
def replace_block(html_text: str, start: str, end: str, inner: str) -> str:
    block = f"{start}\n{inner}\n{end}"
    return re.sub(re.escape(start)+r".*?"+re.escape(end), block, html_text, flags=re.S)

# ===== Renderers =====
def render_article_page(title: str, meta_description: str, hero_repo_rel: str, body_html: str, ts: str, sources_pretty: List[Dict]) -> str:
    with open(ARTICLE_TPL, "r", encoding="utf-8") as f: tpl = Template(f.read())
    return tpl.render(title=title, meta_description=meta_description, hero="../"+hero_repo_rel, body=body_html, timestamp=ts, sources=sources_pretty)

def render_archive(manifest: List[Dict]):
    with open(ARCHIVE_TPL, "r", encoding="utf-8") as f: tpl = Template(f.read())
    articles = [{"href": f"./{m['file']}", "title": m["title"], "image": m.get("image", FALLBACK), "date": m["date"]} for m in manifest]
    html_out = tpl.render(articles=articles)
    os.makedirs(ART_DIR, exist_ok=True)
    with open(os.path.join(ART_DIR, "index.html"), "w", encoding="utf-8") as f: f.write(html_out)

# ===== Main =====
def main():
    os.makedirs(ART_DIR, exist_ok=True)
    interests_weighted = load_interests_weighted()  # List[(interest, weight)]
    manifest  = load_manifest()
    seen_fps  = {a["fingerprint"] for a in manifest if a.get("fingerprint")}

    # memory for dedup
    recent_titles, recent_sigs, recent_domains, token_freq = build_recent_memory(manifest, DEDUP_WINDOW_HOURS)

    # Build candidate list for each interest with multi-source + query packs
    all_candidates: List[Dict] = []
    for interest, weight in interests_weighted:
        if "comet" in interest.lower():
            queries = COMET_QUERIES
        else:
            queries = UFO_QUERIES

        # 1) News search feeds
        for q in queries:
            for it in google_news_rss(q, limit=28):
                all_candidates.append({**it, "interest": interest, "weight": weight})
            for it in bing_news_rss(q, limit=28):
                all_candidates.append({**it, "interest": interest, "weight": weight})

        # 2) Reddit external links
        for sub in REDDIT_SUBS:
            for it in reddit_rss(sub, limit=20):
                all_candidates.append({**it, "interest": interest, "weight": weight})

    # Normalize: drop ads, de-dupe by link host+path
    seen_links: Set[str] = set()
    normed: List[Dict] = []
    for it in all_candidates:
        link = it.get("link","").strip()
        if not link: continue
        try:
            u = urlparse(link)
            key = f"{u.netloc.lower().lstrip('www.')}{u.path}"
        except Exception:
            key = link
        if key in seen_links: continue
        seen_links.add(key)
        normed.append(it)

    # Cutoffs: primary window 24h; backfill to 96h if needed
    cutoff_main    = utcnow() - timedelta(hours=NEWS_LOOKBACK_HOURS)
    cutoff_backfill= utcnow() - timedelta(hours=BACKFILL_LOOKBACK_HOURS)
    primary = [x for x in normed if x["date"] >= cutoff_main]
    backfill= [x for x in normed if cutoff_backfill <= x["date"] < cutoff_main]

    # Score primary first
    def score_row(row: Dict) -> float:
        title = row["title"]; dt = row["date"]
        d = urlparse(row["link"]).netloc.lower().split(":")[0]
        if d.startswith("www."): d = d[4:]
        return score_candidate(title, dt, d, recent_titles, recent_sigs, recent_domains, token_freq, row["weight"])

    ranked = sorted(primary, key=score_row, reverse=True)
    if len(ranked) < NUM_PER_RUN:
        ranked += sorted(backfill, key=score_row, reverse=True)

    used_img_ids: Set[str] = set()
    created = []
    run_tokens: Dict[str,int] = {}
    run_domains: Set[str] = set()
    attempts = 0
    MAX_ATTEMPTS = 150  # allow more passes due to larger candidate pool

    def can_take(title: str) -> bool:
        sig = topic_signature(title)
        if any(run_tokens.get(t,0) >= CAP_PER_TOKEN for t in sig): return False
        for made in created:
            if difflib.SequenceMatcher(None, title.lower(), made["title"].lower()).ratio() >= 0.92:
                return False
        return True

    i = 0
    while len(created) < NUM_PER_RUN and attempts < MAX_ATTEMPTS and i < len(ranked):
        attempts += 1
        cand = ranked[i]; i += 1

        domain = urlparse(cand["link"]).netloc.lower().split(":")[0]
        if domain.startswith("www."): domain = domain[4:]
        if domain in run_domains: continue

        if is_advertorial(cand["title"], cand["link"]): continue
        if not can_take(cand["title"]): continue

        # Get full text (fallback to Playwright in scraper)
        source_text = fetch_page_text(cand["link"], min_chars=250)
        if len(source_text) < 250: continue

        # Strict relevance to the focus interest
        if not passes_interest_relevance(cand["interest"], cand["title"], source_text):
            continue

        # Try to scrape best image from the page
        _scraped_text, scraped_imgs = scrape_article(cand["link"])
        hero_rel = FALLBACK
        if scraped_imgs:
            try:
                hero_rel = download_image(scraped_imgs[0], slugify(cand["title"])[:50])
            except Exception:
                hero_rel = FALLBACK
        if hero_rel == FALLBACK:
            hero_rel = unsplash_unique(cand["interest"], used_img_ids)

        # Write article
        try:
            body_html = openai_article(cand["interest"], cand["title"], [cand["link"]], source_text)
        except Exception as e:
            print("OpenAI error:", e); continue

        filename = f"{slugify(cand['title'])}.html"
        ts = cand["date"].strftime("%Y-%m-%d %H:%M UTC")
        sources_pretty = [{"url": cand["link"], "label": urlparse(cand["link"]).netloc.replace('www.','')}]

        page_html = render_article_page(cand["title"], cand["title"], hero_rel, body_html, ts, sources_pretty)
        with open(os.path.join(ART_DIR, filename), "w", encoding="utf-8") as f: f.write(page_html)

        fp = fingerprint_key(cand["title"], cand["link"])
        manifest.insert(0, {"title": cand["title"], "file": filename, "image": hero_rel, "date": ts, "source_url": cand["link"], "fingerprint": fp})
        save_manifest(manifest)

        created.append(cand)
        for t in topic_signature(cand["title"]):
            run_tokens[t] = run_tokens.get(t,0) + 1
        run_domains.add(domain)

    # Archive + homepage if anything new
    if created:
        render_archive(manifest)

    # homepage blocks
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

    # ticker/trending
    ticker_titles: List[str] = []
    for interest, _w in load_interests_weighted():
        for itm in google_news_rss(interest, limit=2):
            ticker_titles.append(itm["title"])
            if len(ticker_titles) >= TICKER_COUNT: break
        if len(ticker_titles) >= TICKER_COUNT: break
    ticker_text = " · ".join(ticker_titles) if ticker_titles else "Fresh updates every cycle."

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

    # weather (Toronto)
    weather_text = toronto_weather()

    with open(INDEX, "r", encoding="utf-8") as f: idx = f.read()
    idx = replace_block(idx, "<!--SLIDES-->",     "<!--/SLIDES-->",     slides_html)
    idx = replace_block(idx, "<!--FEATURED-->",   "<!--/FEATURED-->",   featured_html)
    idx = replace_block(idx, "<!--HEADLINES-->",  "<!--/HEADLINES-->",  headlines_html)
    idx = replace_block(idx, "<!--TICKER-->",     "<!--/TICKER-->",     ticker_text)
    idx = replace_block(idx, "<!--TRENDING-->",   "<!--/TRENDING-->",   trending_html)
    idx = replace_block(idx, "<!--WEATHER-->",    "<!--/WEATHER-->",    weather_text)
    with open(INDEX, "w", encoding="utf-8") as f: f.write(idx)

# ===== Extras =====
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

def build_recent_memory(manifest: List[Dict], window_hours: int) -> Tuple[List[str], List[Set[str]], Set[str], Dict[str,int]]:
    cut = utcnow() - timedelta(hours=window_hours)
    titles, sigs, domains, token_freq = [], [], set(), {}
    for a in manifest:
        dt = parse_manifest_dt(a.get("date",""))
        if dt and dt >= cut:
            t = a.get("title",""); titles.append(t)
            ts = topic_signature(t); sigs.append(ts)
            for tok in ts: token_freq[tok] = token_freq.get(tok, 0) + 1
            link = a.get("source_url",""); d = urlparse(link).netloc.lower().split(":")[0]
            if d.startswith("www."): d = d[4:]
            if d: domains.add(d)
    return titles, sigs, domains, token_freq

if __name__ == "__main__":
    main()
