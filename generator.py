import os, random, re, requests
from datetime import datetime, timezone
from openai import OpenAI
from jinja2 import Template
from git import Repo

# ────────────── CONFIG ──────────────
INTERESTS    = [
    "AI ethics", "cryptocurrency", "UFO disclosure",
    "telluric currents", "cosmic ray precognition"
]
ARTICLES     = 4
MODEL        = "gpt-4o-mini"        # change to "gpt-4o" if you want higher quality
TOKENS       = 900                  # max tokens per response
OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
UNSPLASH_KEY = os.getenv("UNSPLASH_KEY")
# ─────────────────────────────────────

client = OpenAI(api_key=OPENAI_KEY)

# ─── helper: call GPT ──────────────────────────────────────────────────────────
def chat(prompt: str, max_tokens: int = TOKENS) -> str:
    """Single-shot GPT call with our system prompt."""
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are an extreme-left investigative reporter."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    ).choices[0].message.content.strip()

# ─── helper: build one story ───────────────────────────────────────────────────
def story(topic: str):
    """Return (article_text, list_of_image_prompts)."""
    url   = chat(f"Return ONLY one reputable news URL (<3h old) about {topic}.")
    text  = chat(
        f"Read {url}. Compose an ~800-word article in a punchy, extreme-left tone. "
        "Insert up to 3 image markers like <<<IMG:description>>> where photos belong."
    )

    body_parts = re.split(r'<<<IMG:.*?>>>', text)
    prompts    = re.findall(r'<<<IMG:(.*?)>>>', text)[:3]
    return " ".join(body_parts), prompts

# ─── helper: fetch Unsplash photo ──────────────────────────────────────────────
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

# ─── helper: render article HTML ───────────────────────────────────────────────
def render_article(body: str, imgs: list[str], slug: str):
    with open("templates/article.html.j2", encoding="utf-8") as f:   # UTF-8 force
        template = Template(f.read())

    html = template.render(
        body   = body,
        imgs   = imgs,
        updated= datetime.now(timezone.utc)
    )

    with open(f"articles/{slug}", "w", encoding="utf-8") as f:
        f.write(html)

# ─── helper: inject slides into index.html ─────────────────────────────────────
def update_index(slides_html: str):
    with open("index.html", "r+", encoding="utf-8") as f:
        content = f.read()
        parts   = content.split("<!--SLIDES-->")
        new     = parts[0] + "<!--SLIDES-->\n" + slides_html + "\n<!--SLIDES-->" + parts[2]
        f.seek(0); f.write(new); f.truncate()

# ─── main pipeline ─────────────────────────────────────────────────────────────
def main():
    if not OPENAI_KEY or not UNSPLASH_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY or UNSPLASH_KEY environment variables")

    slides = ""
    for n in range(1, ARTICLES + 1):
        body, img_prompts = story(random.choice(INTERESTS))
        imgs = [grab_img(p) for p in img_prompts]
        slug = f"article{n}.html"

        render_article(body, imgs, slug)

        headline = body.split(".")[0][:120]
        slides  += (
            f'<a class="slide" href="articles/{slug}" '
            f'style="background-image:url(\'{imgs[0]}\')">'
            f'<h2>{headline}</h2></a>\n'
        )

    update_index(slides)

    # auto-commit if there are changes
    repo = Repo(".")
    repo.git.add(all=True)
    if repo.is_dirty():
        repo.index.commit("auto: refresh headlines")
        repo.remote().push()

# ─── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
