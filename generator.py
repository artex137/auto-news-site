import os, re, json, random, datetime, pathlib, textwrap
from typing import List
import requests

# --- OpenAI client (newer SDK name can vary; adapt if needed) ---
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE = pathlib.Path(__file__).parent.resolve()
ART_DIR = BASE / "articles"
ASSETS = BASE / "assets"
TEMPLATE = BASE / "templates" / "article.html.j2"
INDEX = BASE / "index.html"

UNSPLASH_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

INTERESTS = [
  "AI policy and safety", "cryptocurrency regulation", "space exploration", "privacy & surveillance",
  "clean energy breakthroughs", "quantum computing", "UFO/UAP oversight", "global macroeconomics"
]

NUM_ARTICLES = 4

SYSTEM_PERSONA = """
You are **Airyan Fawking**, a millennial, sharp, non‑normie reporter with high pattern‑recognition,
skeptical curiosity, and concise focus. You write **straight news in newspaper style** (AP‑adjacent),
using the **inverted pyramid**: headline → lede (who/what/when/where/why/how) → nut graf →
key facts with quotes and attributions → context → forward‑looking kicker. You avoid editorializing,
snark, or propaganda framing. Voice is lean, vivid, and precise; vocabulary modern but not slangy.
Always include accurate timestamps, clear sourcing, and neutral phrasing.

Output HTML for body paragraphs and subheads (<p>, <h2>, <ul>, <li>, <blockquote> for quotes).
Do not wrap the whole article in <html> or <body>. Keep links as <a>.
"""

URL_FINDER_PROMPT = """
Return exactly one reputable, working news URL (not paywalled if possible) less than 6 hours old
about the topic provided. Respond with ONLY the URL, nothing else.
"""

ARTICLE_PROMPT = """
Write a straight‑news article as described in the system message, based on this source URL:
{url}

Produce a JSON object with:
- "title": a headline (max 90 chars)
- "meta_description": 140–160 characters
- "body_html": HTML with lede first, then subheads and body; include at least one quote with attribution.
- "image_queries": array of 2–3 short queries for illustrative photos (people or concepts from the story)

Facts must reflect the source. Attribute claims to their sources. Include date/time in body.
"""

def ask_url(topic:str)->str:
    msg = [
        {"role":"system","content":URL_FINDER_PROMPT.strip()},
        {"role":"user","content":f"Topic: {topic}"}
    ]
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msg,
        temperature=0.2,
        max_tokens=200
    )
    url = r.choices[0].message.content.strip()
    return url

def write_article(url:str)->dict:
    msg = [
        {"role":"system","content":SYSTEM_PERSONA.strip()},
        {"role":"user","content":ARTICLE_PROMPT.format(url=url).strip()}
    ]
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msg,
        temperature=0.3,
        max_tokens=1600
    )
    text = r.choices[0].message.content
    # Extract JSON (in case model wraps it)
    m = re.search(r'\{.*\}\s*$', text, re.S)
    payload = json.loads(m.group(0) if m else text)
    return payload

def unsplash(query:str)->str:
    if not UNSPLASH_KEY: return ""
    try:
        resp = requests.get(
            "https://api.unsplash.com/photos/random",
            params={"query":query,"orientation":"landscape","content_filter":"high"},
            headers={"Authorization":f"Client-ID {UNSPLASH_KEY}"}, timeout=20
        ).json()
        url = resp.get("urls",{}).get("regular")
        if not url: return ""
        img = requests.get(url, timeout=30).content
        ASSETS.mkdir(exist_ok=True, parents=True)
        fname = f"{resp.get('id','img')}.jpg"
        path = ASSETS / fname
        with open(path,"wb") as f: f.write(img)
        return f"assets/{fname}"
    except Exception:
        return ""

def render_article(payload:dict, images:List[str])->str:
    from jinja2 import Template
    with open(TEMPLATE,"r",encoding="utf-8") as f:
        tpl = Template(f.read())

    title = payload.get("title","Untitled")
    meta = payload.get("meta_description","Latest reporting by Airyan Fawking.")
    body_html = payload.get("body_html","")
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    hero = images[0] if images else ""

    # Ensure paragraphs look good if model returned plain text
    if "<p" not in body_html:
        body_html = "<p>" + body_html.replace("\n\n","</p><p>") + "</p>"

    return tpl.render(
        title=title,
        meta_description=meta,
        body=body_html,
        hero=hero,
        images=images,
        timestamp=ts
    )

def first_sentence(text:str)->str:
    txt = re.sub(r'<[^>]+>', '', text)
    m = re.split(r'(?<=[.!?])\s+', txt.strip(), maxsplit=1)
    return (m[0] if m else txt)[:120]

def update_index(slides_html:str):
    with open(INDEX,"r",encoding="utf-8") as f:
        html = f.read()
    start = html.find("<!--SLIDES-->")
    end = html.find("<!--/SLIDES-->")
    if start==-1 or end==-1:
        # If markers missing, append slides at start of .slides
        html = re.sub(r'(<div class="slides">)(\s*)', r'\1\n' + slides_html + r'\2', html, count=1)
    else:
        new = html[:start] + "<!--SLIDES-->\n" + slides_html + "\n<!--/SLIDES-->" + html[end+13:]
        html = new
    with open(INDEX,"w",encoding="utf-8") as f:
        f.write(html)

def main():
    ART_DIR.mkdir(exist_ok=True, parents=True)

    topics = random.sample(INTERESTS, k=min(NUM_ARTICLES, len(INTERESTS)))
    slide_snippets = []

    for i,topic in enumerate(topics, start=1):
        try:
            url = ask_url(topic)
            payload = write_article(url)

            queries = payload.get("image_queries") or [payload.get("title","news")]
            images = []
            for q in queries[:3]:
                img = unsplash(q)
                if img: images.append(img)
            if not images:
                # one generic fallback gradient card (optional image-less)
                pass

            html = render_article(payload, images)
            # Save article
            art_path = ART_DIR / f"article{i}.html"
            with open(art_path, "w", encoding="utf-8") as f:
                # Pass-through from render_article returns full HTML page
                f.write(html)

            # Build slide (use hero if exists)
            title = payload.get("title","Latest report")
            hero = images[0] if images else ""
            headline = title if title else first_sentence(payload.get("body_html",""))
            style_bg = f" style=\"background-image:url('{hero}')\"" if hero else ""
            slide = (
                f"<a class=\"slide\" href=\"articles/article{i}.html\"{style_bg}>"
                f"<span class=\"badge\">Fresh</span>"
                f"<h2>{headline}</h2>"
                f"</a>"
            )
            slide_snippets.append(slide)

        except Exception as ex:
            print("Error generating article:", ex)
            continue

    update_index("\n".join(slide_snippets))

if __name__ == "__main__":
    main()
