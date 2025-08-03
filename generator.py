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
ARTICLES     = 4
MODEL        = "gpt-4o-mini"  # or "gpt-4o" if you prefer
TOKENS       = 900
OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
UNSPLASH_KEY = os.getenv("UNSPLASH_KEY")
FALLBACK_IMG = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"
# ──────────────────────────────

client = OpenAI(api_key=OPENAI_KEY)

def log(msg: str):
    print(msg)
    sys.stdout.flush()

def chat(prompt: str, max_tokens: int = TOKENS) -> str:
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are an extreme-left investigative reporter."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens
    ).choices[0].message.content.strip()

def story(topic: str):
    url = chat(f"Return ONLY one reputable news URL (<3h old) about {topic}.")
    text = chat(
        f"Read {url}. Write an ~800-word article in a punchy extreme-left tone. "
        "Insert up to 3 image markers like <<<IMG:description>>> where photos belong."
    )
    body_parts = re.split(r'<<<IMG:.*?>>>', text)
    prompts    = re.findall(r'<<<IMG:(.*?)>>>', text)[:3]
    return " ".join(body_parts), prompts

def grab_img(query: str) -> str:
    for attempt in range(1, 4):
        resp = requests.get(
            "https://api.unsplash.com/photos/random",
            params={"query": query, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"}
        )
        if resp.status_code == 200:
            data = resp.json()
            path = f"assets/{data['id']}.jpg"
            with open(path, "wb") as f:
                f.write(requests.get(data["urls"]["regular"]).content)
            log(f"✓ Unsplash image for '{query}'")
            return path
        log(f"Unsplash attempt {attempt}/3 failed ({resp.status_code}) for '{query}'")
    # fallback
    log(f"⚠️  Using fallback image for '{query}'")
    path = "assets/fallback.jpg"
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(requests.get(FALLBACK_IMG).content)
    return path

def render_article(body: str, imgs: list[str], slug: str):
    with open("templates/article.html.j2", encoding="utf-8") as f:
        tpl = Template(f.read())
    html = tpl.render(body=body, imgs=imgs,
                      updated=datetime.now(timezone.utc))
    with open(f"articles/{slug}", "w", encoding="utf-8") as f:
        f.write(html)

def update_index(slides_html: str):
    with open("index.html", "r+", encoding="utf-8") as f:
        head, _, tail = f.read().partition("<!--SLIDES-->")
        _, _, tail    = tail.partition("<!--SLIDES-->")
        f.seek(0)
        f.write(head + "<!--SLIDES-->\n" + slides_html + "\n<!--SLIDES-->" + tail)
        f.truncate()

def main():
    if not OPENAI_KEY or not UNSPLASH_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY or UNSPLASH_KEY")

    slides = ""
    for i in range(1, ARTICLES + 1):
        topic = random.choice(INTERESTS)
        log(f"🔎 Generating article {i}/{ARTICLES}: {topic}")
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
    log("↻ Slides updated in index.html")

    # Stage changes; let the workflow action commit & push
    repo = Repo(".")
    repo.git.add(all=True)
    if repo.is_dirty() and not os.getenv("GITHUB_ACTIONS"):
        repo.index.commit("auto: refresh headlines")
        log("📝 Local commit created")

if __name__ == "__main__":
    main()
