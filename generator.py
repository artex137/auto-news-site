import os
import random
import re
import requests
import sys
from datetime import datetime, timezone
from openai import OpenAI
from jinja2 import Template
from git import Repo

# ─────────── CONFIG ───────────
INTERESTS = [
    "AI ethics",
    "cryptocurrency",
    "UFO disclosure",
    "telluric currents",
    "cosmic ray precognition",
]
ARTICLES = 4
MODEL = "gpt-4o-mini"
TOKENS = 900
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_KEY = os.getenv("UNSPLASH_KEY")
FALLBACK_IMG = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"
# ──────────────────────────────

client = OpenAI(api_key=OPENAI_KEY)

def log(msg):
    print(msg)
    sys.stdout.flush()

def chat(prompt, max_tokens=TOKENS):
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an extreme-left investigative reporter."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens
    ).choices[0].message.content.strip()

def story(topic):
    url = chat(f"Return ONLY one reputable news URL (<3h old) about {topic}.")
    text = chat(
        f"Read {url}. Write an ~800-word article in a punchy extreme-left tone. "
        "Insert up to 3 image markers like <<<IMG:description>>> where photos belong."
    )
    parts   = re.split(r'<<<IMG:.*?>>>', text)
    prompts = re.findall(r'<<<IMG:(.*?)>>>', text)[:3]
    return " ".join(parts), prompts

def grab_img(query):
    for attempt in range(1, 4):
        resp = requests.get(
            "https://api.unsplash.com/photos/random",
            params={"query": query, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"}
        )
        if resp.status_code == 200:
            data = resp.json()
            path = f"assets/{data['id']}.jpg"
            img_data = requests.get(data["urls"]["regular"]).content
            with open(path, "wb") as f:
                f.write(img_data)
            log(f"✓ Unsplash image for '{query}'")
            return path
        log(f"Unsplash {attempt}/3 failed ({resp.status_code}) for '{query}'")
    log(f"⚠️ Using fallback for '{query}'")
    path = "assets/fallback.jpg"
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(requests.get(FALLBACK_IMG).content)
    return path

def render_article(body, imgs, slug):
    with open("templates/article.html.j2", encoding="utf-8") as f:
        tpl = Template(f.read())
    html = tpl.render(body=body, imgs=imgs, updated=datetime.now(timezone.utc))
    with open(f"articles/{slug}", "w", encoding="utf-8") as f:
        f.write(html)

def update_index(slides_html):
    with open("index.html", "r+", encoding="utf-8") as f:
        content = f.read()
        before, sep, rest = content.partition("<!--SLIDES-->")
        middle, sep2, after = rest.partition("<!--SLIDES-->")
        new = before + "<!--SLIDES-->\n" + slides_html + "\n<!--SLIDES-->" + after
        f.seek(0)
        f.write(new)
        f.truncate()

def main():
    if not OPENAI_KEY or not UNSPLASH_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY or UNSPLASH_KEY")

    slides = ""
    for i in range(1, ARTICLES + 1):
        topic = random.choice(INTERESTS)
        log(f"🔎 Article {i}/{ARTICLES}: {topic}")
        body, prompts = story(topic)
        if not prompts:
            prompts = [topic]
        imgs = [grab_img(p) for p in prompts]
        slug = f"article{i}.html"
        render_article(body, imgs, slug)
        headline = body.split(".")[0][:120]
        slides += (
            f'<a class="slide" href="articles/{slug}" '
            f'style="background-image:url(\'{imgs[0]}\')">'
            f'<h2>{headline}</h2></a>\n'
        )

    update_index(slides)
    log("↻ Updated slides in index.html")

    # Stage changes; the workflow action will commit & push them
    repo = Repo(".")
    repo.git.add(all=True)

if __name__ == "__main__":
    main()
