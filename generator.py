import os, re, json, random, datetime, pathlib, unicodedata
from typing import List, Dict
import requests
from openai import OpenAI

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

UNSPLASH_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# 30-topic interest list (broad from your history)
INTERESTS = [
    "cosmic ray influence on precognition",
    "MKSEARCH psi experiments",
    "AI policy and safety",
    "cryptocurrency regulation",
    "Bitcoin market trends",
    "multi-timeframe crypto trading",
    "order flow analysis in markets",
    "Comet 3I/ATLAS trajectory",
    "UFO/UAP government disclosures",
    "sacred geometry in architecture",
    "occult symbolism in politics",
    "quantum computing breakthroughs",
    "esoteric numerology and 137",
    "AI in automated journalism",
    "high-altitude particle experiments",
    "free energy research developments",
    "space exploration missions",
    "clean energy breakthroughs",
    "metaphysical event predictions",
    "geopolitical tensions in Asia",
    "privacy & surveillance policy",
    "off-grid living in Alaska",
    "predictive programming in media",
    "ancient measurement systems",
    "lunar and solar cycles",
    "stock market algorithmic trading",
    "deep sea exploration news",
    "volcanic activity and climate",
    "AI safety & governance",
    "macro economics and rates"
]

NUM_ARTICLES = 4         # per run
SLIDER_LATEST = 4        # how many to show on homepage
FALLBACK = "assets/fallback-hero.jpg"  # ensure this exists

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

# ===== Utilities =====
def slugify(value:str, max_len:int=60)->str:
    value = unicodedata.normalize('NFKD', value)
    value = re.sub(r'[^\w\s-]', '', value, flags=re.U).strip().lower()
    value = re.sub(r'[\s_-]+', '-', value)
    return value[:max_len].strip('-') or "article"

def ask_url(topic:str)->str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":URL_FINDER},
                  {"role":"user","content":topic}],
        temperature=0.2, max_tokens=200
    )
    return r.choices[0].message.content.strip()

def write_article(url:str)->Dict:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":SYSTEM_STYLE},
                  {"role":"user","content":ARTICLE_PROMPT.format(url=url)}],
        temperature=0.3, max_tokens=1600
    )
    text = r.choices[0].message.content.strip()
    m = re.search(r'\{.*\}\s*$', text, re.S)
    return json.loads(m.group(0) if m else text)

def unsplash(query:str)->str:
    if not UNSPLASH_KEY:
        return FALLBACK
    try:
        resp = requests.get(
            "https://api.unsplash.com/photos/random",
            params={"query":query,"orientation":"landscape","content_filter":"high"},
            headers={"Authorization":f"Client-ID {UNSPLASH_KEY}"},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        url = data.get("urls",{}).get("regular")
        if not url:
            return FALLBACK
        img_bytes = requests.get(url, timeout=20).content
        ASSETS.mkdir(exist_ok=True, parents=True)
        fname = f"{data.get('id','img')}.jpg"
        path = ASSETS / fname
        with open(path,"wb") as f:
            f.write(img_bytes)
        return f"assets/{fname}"
    except Exception as e:
        print("Unsplash error:", e)
        return FALLBACK

def render_article(payload:Dict, images:List[str], ts_utc:str)->str:
    from jinja2 import Template
    with open(ARTICLE_TPL,"r",encoding="utf-8") as f:
        tpl = Template(f.read())
    hero = images[0] if images else FALLBACK
    return tpl.render(
        title=payload.get("title","Untitled"),
        meta_description=payload.get("meta_description",""),
        body=payload.get("body_html",""),
        hero=hero,
        images=images,
        timestamp=ts_utc
    )

def update_index(slides_html:str):
    with open(INDEX,"r",encoding="utf-8") as f:
        html = f.read()
    block = f"<!--SLIDES-->\n{slides_html}\n<!--/SLIDES-->"
    html = re.sub(r'<!--SLIDES-->.*?<!--/SLIDES-->', block, html, flags=re.S)
    with open(INDEX,"w",encoding="utf-8") as f:
        f.write(html)

def render_archive(manifest:List[Dict]):
    from jinja2 import Template
    with open(ARCHIVE_TPL,"r",encoding="utf-8") as f:
        tpl = Template(f.read())
    # Newest first
    articles = [{
        "href": f"./{m['file']}",
        "title": m["title"],
        "image": m.get("image", FALLBACK),
        "date": m["date"]
    } for m in sorted(manifest, key=lambda x: x["date"], reverse=True)]
    out = tpl.render(articles=articles)
    (ART_DIR).mkdir(exist_ok=True, parents=True)
    with open(ART_DIR / "index.html","w",encoding="utf-8") as f:
        f.write(out)

def load_manifest()->List[Dict]:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_manifest(items:List[Dict]):
    MANIFEST.write_text(json.dumps(items, indent=2), encoding="utf-8")

# ===== Main =====
def main():
    ART_DIR.mkdir(exist_ok=True, parents=True)
    manifest = load_manifest()

    created = []
    topics = random.sample(INTERESTS, k=min(NUM_ARTICLES, len(INTERESTS)))

    for topic in topics:
        try:
            url = ask_url(topic)
            payload = write_article(url)
            title = payload.get("title","Untitled").strip()
            slug = slugify(title)
            ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

            # Images
            queries = payload.get("image_queries") or [title]
            images = []
            for q in queries[:3]:
                img = unsplash(q)
                if img: images.append(img)
            if not images: images = [FALLBACK]

            # Render + save article (permanent)
            html = render_article(payload, images, ts)
            filename = f"{slug}.html"
            with open(ART_DIR / filename, "w", encoding="utf-8") as f:
                f.write(html)

            item = {
                "title": title,
                "file": filename,
                "image": images[0],
                "date": ts
            }
            manifest.append(item)
            created.append(item)

        except Exception as e:
            print("Error generating:", e)

    # Persist manifest and rebuild archive
    if created:
        save_manifest(manifest)
        render_archive(manifest)

        # Update homepage slider with latest N
        latest = sorted(manifest, key=lambda x: x["date"], reverse=True)[:SLIDER_LATEST]
        slides = []
        for m in latest:
            slides.append(
                f"<a class=\"slide\" href=\"articles/{m['file']}\" style=\"background-image:url('{m['image']}')\">"
                f"<div class=\"slide-content\"><h2 class=\"slide-headline\">{m['title']}</h2></div>"
                f"</a>"
            )
        update_index("\n".join(slides))
    else:
        # Keep homepage as-is if nothing was created
        pass

if __name__ == "__main__":
    main()
