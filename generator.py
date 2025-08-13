import os
import json
import requests
import re
import html
import hashlib
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader

# === CONFIG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_KEY = os.getenv("UNSPLASH_KEY")

INTERESTS = [
    "911 theories",
    "false flags",
    "aliens",
    "comet 3i atlas",
    "ufos",
    "joe rogan podcasts",
    "president trump"
]

ARTICLE_MIN_WORDS = 800
NEWS_LOOKBACK_HOURS = 24
DATA_FILE = "data/articles.json"
ASSETS_DIR = "assets"

# === SETUP ===
env = Environment(loader=FileSystemLoader("templates"))
os.makedirs("articles", exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# === HELPERS ===
def slugify(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

def fingerprint(title, url):
    return hashlib.md5((title.lower().strip() + url.lower().strip()).encode()).hexdigest()

def load_existing_fingerprints():
    if not os.path.exists(DATA_FILE):
        return set()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {a["fingerprint"] for a in data}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def fetch_google_news(topic):
    rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(topic)}&hl=en-CA&gl=CA&ceid=CA:en"
    r = requests.get(rss_url, timeout=10)
    r.raise_for_status()
    return r.text

def parse_rss(xml_text):
    matches = re.findall(r"<item>.*?<title><!\[CDATA\[(.*?)\]\]></title>.*?<link>(.*?)</link>.*?<pubDate>(.*?)</pubDate>", xml_text, re.S)
    articles = []
    for title, link, pub_date in matches:
        dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
        articles.append({"title": html.unescape(title), "link": link, "date": dt})
    return articles

def fetch_unsplash_image(query):
    if not UNSPLASH_KEY:
        return "assets/fallback-hero.jpg"
    url = "https://api.unsplash.com/photos/random"
    params = {"query": query, "orientation": "landscape", "content_filter": "high"}
    headers = {"Authorization": f"Client-ID {UNSPLASH_KEY}"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    if r.status_code != 200:
        return "assets/fallback-hero.jpg"
    data = r.json()
    img_url = data.get("urls", {}).get("regular")
    if not img_url:
        return "assets/fallback-hero.jpg"
    img_data = requests.get(img_url, timeout=10).content
    fname = f"{slugify(query)}-{data.get('id')}.jpg"
    path = os.path.join(ASSETS_DIR, fname)
    with open(path, "wb") as f:
        f.write(img_data)
    return f"{ASSETS_DIR}/{fname}"

def generate_article_content(topic, source_text):
    prompt = (
        f"Write an approximately {ARTICLE_MIN_WORDS} word newspaper-style news article "
        f"based on the following recent information, but in the tone of a crime desk reporter "
        f"who is also a skeptical conspiracy theorist. Maintain journalistic structure, "
        f"clear paragraphs, and factual reporting style, but let subtle skepticism about official "
        f"narratives show through in wording and detail selection.\n\n"
        f"Topic: {topic}\n\n"
        f"Source info:\n{source_text}\n"
    )
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        timeout=60
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return content

def fetch_trending_terms():
    xml_text = fetch_google_news("trending")
    items = parse_rss(xml_text)
    terms = list({re.sub(r'[^A-Za-z0-9 ]+', '', a["title"]).strip() for a in items})[:10]
    return terms

def fetch_weather_toronto():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": 43.7, "longitude": -79.42, "current_weather": True}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json().get("current_weather", {})
    return f"{data.get('temperature')}°C, {data.get('windspeed')} km/h wind" if data else None

# === MAIN ===
def main():
    existing_fps = load_existing_fingerprints()
    all_articles = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            all_articles = json.load(f)

    cutoff = datetime.utcnow() - timedelta(hours=NEWS_LOOKBACK_HOURS)
    new_items = []

    for topic in INTERESTS:
        xml_text = fetch_google_news(topic)
        for art in parse_rss(xml_text):
            if art["date"] < cutoff:
                continue
            fp = fingerprint(art["title"], art["link"])
            if fp in existing_fps:
                continue
            img_path = fetch_unsplash_image(topic)
            try:
                body = generate_article_content(topic, f"{art['title']} - {art['link']}")
            except Exception as e:
                print(f"OpenAI error: {e}")
                continue
            filename = f"{slugify(art['title'])}.html"
            file_path = os.path.join("articles", filename)
            tmpl = env.get_template("article.html.j2")
            html_content = tmpl.render(
                title=art["title"],
                meta_description=art["title"],
                hero=f"../{img_path}",
                body=body,
                timestamp=art["date"].strftime("%B %d, %Y"),
            )
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            entry = {
                "title": art["title"],
                "href": f"articles/{filename}",
                "image": img_path,
                "date": art["date"].strftime("%b %d, %Y"),
                "file": filename,
                "fingerprint": fp
            }
            all_articles.insert(0, entry)
            existing_fps.add(fp)
            new_items.append(entry)

    save_data(all_articles)

    # Homepage render
    trending_terms = fetch_trending_terms()
    weather = fetch_weather_toronto()
    latest = all_articles[:8]
    main_slider = latest[:5]

    slider_html = "\n".join(
        f"<a class='slide' href='articles/{m['file']}' style=\"background-image:url('{m['image']}')\">"
        f"<div class='slide-content'><h2 class='slide-headline'>{html.escape(m['title'])}</h2></div></a>"
        for m in main_slider
    )

    tmpl_home = env.get_template("index.html")
    home_html = tmpl_home.render(
        slider_html=slider_html,
        latest_articles=latest,
        trending_terms=trending_terms,
        weather=weather,
        generated=datetime.utcnow().strftime("%B %d, %Y %H:%M UTC"),
    )
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(home_html)

if __name__ == "__main__":
    main()
