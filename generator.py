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
    from readability import Document  # readability-lxml
except Exception:
    Document = None
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
STORY_MEM_FILE = os.path.join(DATA_DIR, "story_memory.json")

TPL_DIR       = os.path.join(BASE, "templates")
ARTICLE_TPL   = os.path.join(TPL_DIR, "article.html.j2")
ARCHIVE_TPL   = os.path.join(TPL_DIR, "articles_index.html.j2")
INDEX         = os.path.join(BASE, "index.html")

DEFAULT_INTERESTS = [
    "911 theories","false flags","aliens","comet 3i atlas","ufos","joe rogan podcasts","president trump",
]

NEWS_LOOKBACK_HOURS = 24
DEDUP_WINDOW_HOURS  = 72
STORY_COOLDOWN_HRS  = 6

NUM_PER_RUN    = 4
SLIDER_LATEST  = 4
FEATURED_COUNT = 6
HEADLINES_COUNT= 10
TRENDING_COUNT = 8
TICKER_COUNT   = 12
FALLBACK       = "assets/fallback-hero.jpg"

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

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
UA_GOOGLEBOT = {
    "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Accept-Language": "en-US,en;q=0.9",
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
    if changed: save_manifest(data)
    return data

def save_manifest(items: List[Dict]):
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

# ---- Story memory (cooldown + context) ----
def load_story_memory() -> Dict[str, Dict]:
    if not os.path.exists(STORY_MEM_FILE):
        return {}
    try:
        with open(STORY_MEM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_story_memory(mem: Dict[str, Dict]):
    if len(mem) > 180:
        items = sorted(mem.items(), key=lambda kv: kv[1].get("last_ts", ""), reverse=True)[:150]
        mem = dict(items)
    with open(STORY_MEM_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2, ensure_ascii=False)

# ===== Source handling =====
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
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
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
        print("RSS error:", e)
        return []

# ===== Robust article fetching & extraction =====
def first_external_link(soup: BeautifulSoup) -> Optional[str]:
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = urljoin("https://news.google.com", href)
        if href.startswith("http"):
            d = urlparse(href).netloc.lower()
            if d.startswith("www."): d = d[4:]
            if "news.google.com" not in d and "consent.google.com" not in d:
                return href
    return None

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
                if body and isinstance(body, str) and len(body) > 300:
                    return re.sub(r"\s+", " ", body).strip()
    return None

def extract_main_text(soup: BeautifulSoup) -> str:
    art = soup.find("article")
    if art:
        txt = " ".join(p.get_text(" ", strip=True) for p in art.find_all(["p","li"]))
        if len(txt) > 400:
            return re.sub(r"\s+", " ", txt).strip()
    candidates = []
    for sel in ["main","div","section"]:
        for el in soup.find_all(sel):
            cls = " ".join(el.get("class", [])); idv = el.get("id","")
            score = 0
            if any(k in (cls or "").lower() for k in ["content","article","story","post","entry","body","read"]): score += 2
            if any(k in (idv or "").lower() for k in ["content","article","story","post","entry","body","read"]): score += 2
            score += min(len(el.find_all("p")), 8)
            if score >= 5:
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

def fetch_page_text(url: str, title: Optional[str] = None, min_chars: int = 600, max_chars: int = 7000) -> str:
    """
    Fetch & extract article:
      1) Follow Google News to publisher
      2) JSON-LD articleBody -> <article> -> heuristics
      3) <link rel='amphtml'>, '/amp', '?output=amp' fallbacks
      4) Readability
      5) Retry with Googlebot UA
      6) If still short and title given, find an alternate publisher via Google News
    """
    tried_urls = set()
    def _extract_from_html(u: str, html_text: str) -> str:
        if BeautifulSoup:
            soup = BeautifulSoup(html_text, "lxml")
            # try canonical fetch if different domain
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
            # AMP via rel=amphtml
            if len(body) < min_chars:
                amp_link = soup.find("link", rel=lambda v: v and "amphtml" in v.lower())
                if amp_link and amp_link.get("href"):
                    au = urljoin(u, amp_link["href"])
                    if au not in tried_urls:
                        try:
                            u3, t3 = fetch_html(au, UA_HEADERS)
                            tried_urls.add(u3)
                            if BeautifulSoup:
                                soup3 = BeautifulSoup(t3, "lxml")
                                body2 = try_jsonld_article(soup3) or extract_main_text(soup3)
                                if Document and len(body2) < min_chars:
                                    via_read2 = extract_with_readability(t3)
                                    if len(via_read2) > len(body2):
                                        body2 = via_read2
                            else:
                                body2 = re.sub("<[^>]+>", " ", t3)
                            if len(body2) > len(body):
                                body = body2
                        except Exception:
                            pass
            return body
        else:
            body = re.sub("<script.*?</script>", " ", html_text, flags=re.S|re.I)
            body = re.sub("<style.*?</style>",  " ", body, flags=re.S|re.I)
            body = re.sub("<[^>]+>", " ", body)
            return re.sub(r"\s+", " ", body).strip()

    try:
        # 1) initial
        final_url, html_text = fetch_html(url, UA_HEADERS)
        tried_urls.add(final_url)
        # handle Google News interstitial
        if "news.google.com" in urlparse(final_url).netloc.lower():
            if BeautifulSoup:
                soup = BeautifulSoup(html_text, "lxml")
                # meta refresh or first external link
                meta = soup.find("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)})
                next_url = None
                if meta and "url=" in (meta.get("content") or "").lower():
                    m = re.search(r'url=([^;]+)', meta.get("content"), flags=re.I)
                    if m:
                        cand = html.unescape(m.group(1).strip().strip("'\""))
                        next_url = cand if cand.startswith("http") else urljoin(final_url, cand)
                if not next_url:
                    next_url = first_external_link(soup) or final_url
                if next_url != final_url and next_url not in tried_urls:
                    final_url, html_text = fetch_html(next_url, UA_HEADERS)
                    tried_urls.add(final_url)

        body = _extract_from_html(final_url, html_text)

        # 2) /amp or ?output=amp
        if len(body) < min_chars:
            amp_try = None
            if final_url.endswith("/"): amp_try = final_url + "amp"
            else: amp_try = final_url + "/amp"
            try:
                u3, t3 = fetch_html(amp_try, UA_HEADERS)
                tried_urls.add(u3)
                if BeautifulSoup:
                    soup3 = BeautifulSoup(t3, "lxml")
                    body2 = try_jsonld_article(soup3) or extract_main_text(soup3)
                    if Document and len(body2) < min_chars:
                        via_read3 = extract_with_readability(t3)
                        if len(via_read3) > len(body2):
                            body2 = via_read3
                else:
                    body2 = re.sub("<[^>]+>", " ", t3)
                if len(body2) > len(body):
                    body = body2
            except Exception:
                pass
        if len(body) < min_chars:
            try:
                u4, t4 = fetch_html(final_url + "?output=amp", UA_HEADERS)
                tried_urls.add(u4)
                if BeautifulSoup:
                    soup4 = BeautifulSoup(t4, "lxml")
                    body3 = try_jsonld_article(soup4) or extract_main_text(soup4)
                    if Document and len(body3) < min_chars:
                        via_read4 = extract_with_readability(t4)
                        if len(via_read4) > len(body3):
                            body3 = via_read4
                else:
                    body3 = re.sub("<[^>]+>", " ", t4)
                if len(body3) > len(body):
                    body = body3
            except Exception:
                pass

        # 3) Googlebot UA retry
        if len(body) < min_chars:
            try:
                u5, t5 = fetch_html(final_url, UA_GOOGLEBOT)
                tried_urls.add(u5)
                if BeautifulSoup:
                    soup5 = BeautifulSoup(t5, "lxml")
                    body4 = try_jsonld_article(soup5) or extract_main_text(soup5)
                    if Document and len(body4) < min_chars:
                        via_read5 = extract_with_readability(t5)
                        if len(via_read5) > len(body4):
                            body4 = via_read5
                else:
                    body4 = re.sub("<[^>]+>", " ", t5)
                if len(body4) > len(body):
                    body = body4
            except Exception:
                pass

        # 4) Alternate publisher by title
        if len(body) < min_chars and title:
            alts = google_news_rss(title, limit=6)
            for alt in alts:
                alt_link = alt["link"]
                if alt_link in tried_urls: continue
                if is_advertorial(title, alt_link): continue
                try:
                    u6, t6 = fetch_html(alt_link, UA_HEADERS)
                    tried_urls.add(u6)
                    if BeautifulSoup:
                        soup6 = BeautifulSoup(t6, "lxml")
                        body5 = try_jsonld_article(soup6) or extract_main_text(soup6)
                        if Document and len(body5) < min_chars:
                            via_read6 = extract_with_readability(t6)
                            if len(via_read6) > len(body5):
                                body5 = via_read6
                    else:
                        body5 = re.sub("<[^>]+>", " ", t6)
                    if len(body5) >= min_chars:
                        body = body5
                        break
                except Exception:
                    continue

        body = re.sub(r"\s+", " ", body or "").strip()[:max_chars]
        return body
    except Exception as e:
        print("parse error:", e)
        return ""

# ===== Similarity / clustering / scoring (unchanged core logic) =====
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

def cluster_key_from_title(title: str) -> str:
    sig = topic_signature(title)
    return "-".join(sorted(sig)) or slugify(title)[:24]

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

def normalize_context(txt: str) -> str:
    t = re.sub(r"\s+", " ", (txt or "").strip().lower())
    t = re.sub(r"[^a-z0-9\.\,\:\;\-\(\)\/\s]", " ", t)
    return re.sub(r"\s+", " ", t)

def numbers_set(txt: str) -> Set[str]:
    return set(re.findall(r"\b\d[\d,\.]*\b", txt or ""))

def significant_update(new_text: str, old_text: str) -> bool:
    if not old_text: return True
    a = normalize_context(old_text)
    b = normalize_context(new_text)
    if not b: return False
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    new_nums = numbers_set(b) - numbers_set(a)
    len_change = abs(len(b) - len(a)) / max(1, len(a))
    return (ratio <= 0.75) or (len(new_nums) >= 1) or (len_change >= 0.20)

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
    with open(os.path.join(ART_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_out)

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

# ===== Selection logic =====
def score_candidate(title: str, dt: datetime, domain: str, recent_titles: List[str], recent_sigs: List[Set[str]], recent_domains: Set[str], token_freq: Dict[str,int]) -> float:
    r = recency_score(dt)
    n = novelty_score(title, recent_titles, recent_sigs)
    dom_penalty = 0.12 if domain in recent_domains else 0.0
    sig = topic_signature(title)
    over = max((token_freq.get(t,0) for t in sig), default=0)
    tok_penalty = min(0.20, 0.04 * over)
    return 0.55*r + 0.35*n - dom_penalty - tok_penalty

def pick_round_robin(cands_by_interest: Dict[str, List[Dict]], k: int) -> List[Dict]:
    chosen: List[Dict] = []
    run_tokens: Dict[str,int] = {}
    interests = [i for i in cands_by_interest.keys() if cands_by_interest[i]]
    idx = 0
    CAP_PER_TOKEN = 1
    while len(chosen) < k and interests:
        i = interests[idx % len(interests)]
        pool = cands_by_interest[i]
        while pool and (pool[0].get("_blocked", False)):
            pool.pop(0)
        if not pool:
            interests.remove(i)
            if not interests: break
            continue
        cand = pool.pop(0)
        sig = topic_signature(cand["title"])
        if any(run_tokens.get(t,0) >= CAP_PER_TOKEN for t in sig):
            cand["_blocked"] = True
            pool.insert(0, cand)
            idx += 1
            continue
        chosen.append(cand)
        for t in sig:
            run_tokens[t] = run_tokens.get(t,0) + 1
        idx += 1
    if len(chosen) < k:
        leftovers = []
        for plist in cands_by_interest.values():
            leftovers.extend([c for c in plist if not c.get("_blocked")])
        for c in leftovers:
            if len(chosen) >= k: break
            sig = topic_signature(c["title"])
            if any(run_tokens.get(t,0) >= CAP_PER_TOKEN for t in sig):
                continue
            chosen.append(c)
            for t in sig:
                run_tokens[t] = run_tokens.get(t,0) + 1
    return chosen[:k]

# ===== Main =====
def main():
    os.makedirs(ART_DIR, exist_ok=True)
    interests = load_interests()
    manifest  = load_manifest()
    story_mem = load_story_memory()
    seen_fps  = {a["fingerprint"] for a in manifest if a.get("fingerprint")}
    cutoff    = utcnow() - timedelta(hours=NEWS_LOOKBACK_HOURS)

    recent_titles, recent_sigs, recent_domains, token_freq = build_recent_memory(manifest, DEDUP_WINDOW_HOURS)

    all_candidates: List[Dict] = []
    for topic in interests:
        feed = google_news_rss(topic, limit=16)
        for it in feed:
            it_date = as_utc(it["date"])
            if it_date < cutoff: continue
            title = it["title"]; link = it["link"]
            fp = fingerprint_key(title, link)
            if fp in seen_fps: continue
            if is_advertorial(title, link): continue
            if is_same_story(title, recent_titles, recent_sigs): continue

            cluster = cluster_key_from_title(title)
            mem = story_mem.get(cluster)
            src_text = None
            if mem:
                try:
                    last_ts = datetime.fromisoformat(mem.get("last_ts")).replace(tzinfo=timezone.utc)
                except Exception:
                    last_ts = None
                cooldown_active = last_ts and (utcnow() - last_ts) < timedelta(hours=STORY_COOLDOWN_HRS)
                if cooldown_active:
                    src_text = fetch_page_text(link, title=title)
                    if not significant_update(src_text, mem.get("last_context","")):
                        continue

            dom = urlparse(link).netloc.lower().split(":")[0]
            if dom.startswith("www."): dom = dom[4:]
            score = score_candidate(title, it_date, dom, recent_titles, recent_sigs, recent_domains, token_freq)
            all_candidates.append({
                "topic": topic, "title": title, "link": link, "date": it_date, "domain": dom,
                "score": score, "cluster": cluster, "source_text": src_text
            })

    cands_by_interest: Dict[str, List[Dict]] = {}
    for topic in interests:
        group = [c for c in all_candidates if c["topic"] == topic]
        group.sort(key=lambda x: x["score"], reverse=True)
        cands_by_interest[topic] = group

    uniq = pick_round_robin(cands_by_interest, NUM_PER_RUN)

    used_img_ids: Set[str] = set()
    created: List[Dict] = []

    for c in uniq:
        source_text = c.get("source_text") or fetch_page_text(c["link"], title=c["title"])
        if not source_text or len(source_text) < 600:
            print("Skipped (insufficient source text):", c["title"])
            continue

        try:
            body_html = openai_article(c["topic"], c["title"], c["link"], source_text)
        except Exception as e:
            print("OpenAI error:", e); continue

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
        with open(os.path.join(ART_DIR, filename), "w", encoding="utf-8") as f:
            f.write(page_html)

        fp = fingerprint_key(c["title"], c["link"])
        manifest.insert(0, {"title": c["title"], "file": filename, "image": img_repo_rel, "date": ts, "source_url": c["link"], "fingerprint": fp})
        created.append(c)

        recent_titles.append(c["title"])
        recent_sigs.append(topic_signature(c["title"]))

        story_mem[c["cluster"]] = {
            "last_ts": utcnow().isoformat(),
            "last_title": c["title"],
            "last_context": (source_text or "")[:3000],
        }

    if created:
        save_manifest(manifest)
        render_archive(manifest)
        save_story_memory(story_mem)

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

    ticker_titles: List[str] = []
    for topic in interests:
        for itm in google_news_rss(topic, limit=2):
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
