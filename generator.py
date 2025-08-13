import os
import json
import requests
import hashlib
import random
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader
from bs4 import BeautifulSoup

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UNSPLASH_KEY = os.getenv("UNSPLASH_KEY")

ARTICLES_FILE = "articles.json"
ASSETS_DIR = "assets"
TEMPLATES_DIR = "templates"

INTERESTS = [
    "911 theories", "false flags", "aliens", "comet 3i atlas", "ufos",
    "joe rogan podcasts", "president trump"
]

env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

def load_existing_fingerprints():
    if not os.path.exists(ARTICLES_FILE):
        return set()
    with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return {a.get("fingerprint") for a in data if "fingerprint" in a}
        except json.JSONDecodeError:
            return set()

def save_articles(articles):
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

def fetch_news(interest):
    url = f"https://news.google.com/rss/search?q={interest.replace(' ', '%20')}&hl=en-CA&gl=CA&ceid=CA:en"
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "xml")
    items = soup.find_all("item")
    now = datetime.utcnow()
    news = []
    for item in items:
        try:
            pub = datetime.strptime(item.pubDate.text, "%a, %d %b %Y %H:%M:%S %Z")
        except:
            continue
        if now - pub > timedelta(hours=24):
            continue
        link = item.link.text.strip()
        title = item.title.text.strip()
        fp = hashlib.sha256(link.encode()).hexdigest()
        news.append({"title": title, "link": link, "fingerprint": fp})
    return news

def choose_image(query, used_images):
    url = f"https://api.unsplash.com/photos/random?query={query}&orientation=landscape&client_id={UNSPLASH_KEY}"
    for _ in range(5):
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            continue
        data = resp.json()
        img_url = data.get("urls", {}).get("regular")
        if img_url and img_url not in used_images:
            used_images.add(img_url)
            filename = os.path.join(ASSETS_DIR, f"{hashlib.sha256(img_url.encode()).hexdigest()}.jpg")
            with open(filename, "wb") as f:
                f.write(requests.get(img_url).content)
            return filename
    return "assets/fallback-hero.jpg"

def generate_article_text(title, link):
    prompt = (
        f"You are 'Airyan Fawking', a millennial crime-desk reporter with a sharp, skeptical mind "
        f"and a conspiratorial edge. Write an ~800-word news article based strictly on this real news source, "
        f"with detailed reporting and subtle hints at hidden agendas:\n\nTitle: {title}\nSource: {link}"
    )
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
    )
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

def fetch_weather():
    url = "https://api.open-meteo.com/v1/forecast?latitude=43.7&longitude=-79.42&current_weather=true"
    resp = requests.get(url, timeout=10).json()
    cw = resp.get("current_weather", {})
    return f"{cw.get('temperature', '?')}°C, {cw.get('windspeed', '?')} km/h wind"

def fetch_trending_terms():
    feed = "https://trends.google.com/trends/trendingsearches/daily/rss?geo=CA"
    soup = BeautifulSoup(requests.get(feed, timeout=10).text, "xml")
    return [item.title.text for item in soup.find_all("item")[:10]]

def fetch_ticker():
    feed = "https://news.google.com/rss?hl=en-CA&gl=CA&ceid=CA:en"
    soup = BeautifulSoup(requests.get(feed, timeout=10).text, "xml")
    return [item.title.text for item in soup.find_all("item")[:10]]

def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    existing_fps = load_existing_fingerprints()
    try:
        with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
            articles = json.load(f)
    except:
        articles = []

    used_images = set(a["image_url"] for a in articles if "image_url" in a)
    new_articles = []

    for interest in INTERESTS:
        news_items = fetch_news(interest)
        for n in news_items:
            if n["fingerprint"] in existing_fps:
                continue
            img = choose_image(interest, used_images)
            body = generate_article_text(n["title"], n["link"])
            file_slug = f"{hashlib.sha256(n['title'].encode()).hexdigest()}.html"
            new_articles.append({
                "title": n["title"],
                "file": file_slug,
                "image_url": img,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "fingerprint": n["fingerprint"],
                "summary": body[:200] + "...",
                "body": body
            })
            existing_fps.add(n["fingerprint"])
            if len(new_articles) >= 4:
                break
        if len(new_articles) >= 4:
            break

    if new_articles:
        articles = new_articles + articles
        save_articles(articles)

    weather = fetch_weather()
    trending = fetch_trending_terms()
    ticker = fetch_ticker()

    # Render index.html
    idx_tpl = env.get_template("index.html")
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(idx_tpl.render(
            weather=weather, trending_terms=trending, ticker_items=ticker, articles=articles[:6]
        ))

    # Render articles index
    ai_tpl = env.get_template("articles_index.html.j2")
    with open("articles_index.html", "w", encoding="utf-8") as f:
        f.write(ai_tpl.render(articles=articles, trending_terms=trending))

    # Render individual articles
    art_tpl = env.get_template("article.html.j2")
    for a in new_articles:
        with open(os.path.join("articles", a["file"]), "w", encoding="utf-8") as f:
            f.write(art_tpl.render(
                title=a["title"],
                meta_description=a["summary"],
                timestamp=a["timestamp"],
                hero=a["image_url"],
                body=a["body"],
                trending_terms=trending
            ))

if __name__ == "__main__":
    main()
