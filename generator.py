import os, random, re, requests
from datetime import datetime, timezone
from openai import OpenAI
from jinja2 import Template
from git import Repo

# ─────────── CONFIG ───────────
INTERESTS = [
    "AI ethics", "cryptocurrency", "UFO disclosure",
    "telluric currents", "cosmic ray precognition"
]
ARTICLES     = 4
MODEL        = "gpt-4o-mini"        # change to "gpt-4o" for higher quality
TOKENS       = 900
OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
UNSPLASH_KEY = os.getenv("UNSPLASH_KEY")
# ──────────────────────────────

client = OpenAI(api_key=OPENAI_KEY)

# ─── GPT helper ─────────────────
def chat(prompt: str, max_tokens: int = TOKENS) -> str:
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are an extreme-left investigative reporter."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    ).choices[0].message.content.strip()

# ─── Build one story ────────────
def story(topic: str):
    url  = chat(f"Return ONLY one reputable news URL (<3h old) about {topic}.")
    text = chat(
        f"Read {url}. Write an ~800-word article in a punchy extreme-left tone. "
        "Insert up to 3 image markers like <<<IMG:description>>> where photos belong."
    )
    body_parts = re.split(r'<<<IMG:.*?>>>', text)
    prompts    = re.findall(r'<<<IMG:(.*?)>>>', text)[:3]
    return " ".join(body_parts), prompts

# ─── Fetch Unsplash photo ───────
def grab_img(query: str) -> str:
    r = requests.get(
        "https://api.unsplash.com/photos/random",
        params={"query": query, "orientation": "landscape"},
        headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"}
    ).json()
    path = f"assets/{r['id']}.jpg"
    with open(path, "wb") as f:
        f.write(requests.get(r["urls"]["regular"]).content)
    return path

# ─── Render article HTML ────────
def render_article(body: str, imgs: list[str], slug: str):
    with open("templates/article.html.j2", encoding="utf-8") as f:
        template = Template(f.read())
    html = template.render(body=body, imgs=imgs,
                           updated=datetime.now(timezone.utc))
    with open(f"articles/{slug}", "w", encoding="utf-8") as f:
        f.write(html)

# ─── Replace slides in index.html
def update_index(slides_html: str):
    with open("index.html", "r+", encoding="utf-8") as f:
        content = f.read().split("<!--SLIDES-->")
        new = content[0] + "<!--SLIDES-->\n" + slides_html + "\n<!--SLIDES-->" + content[2]
        f.seek(0); f.write(new); f.truncate()

# ─── Main pipeline ──────────────
def main():
    if not OPENAI_KEY or not UNSPLASH_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY or UNSPLASH_KEY")

    slides = ""
    for n in range(1, ARTICLES + 1):
        body, img_prompts = story(random.choice(INTERESTS))
        imgs = [grab_img(p) for p in img_prompts]
        slug = f"article{n}.html"
        render_article(body, imgs, slug)

        headline = body.split(".")[0][:120]
        slides += (
            f'<a class="slide" href="articles/{slug}" '
            f'style="background-image:url(\'{imgs[0]}\')">'
            f'<h2>{headline}</h2></a>\n'
        )

    update_index(slides)

    # Commit locally; skip push when inside GitHub Actions
    repo = Repo(".")
    repo.git.add(all=True)
    if repo.is_dirty():
        repo.index.commit("auto: refresh headlines")
        if not os.getenv("GITHUB_ACTIONS"):
            repo.remote().push()

if __name__ == "__main__":
    main()
