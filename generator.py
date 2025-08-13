import os, re, json, random, datetime, pathlib
from typing import List
import requests
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE = pathlib.Path(__file__).parent.resolve()
ART_DIR = BASE / "articles"
ASSETS = BASE / "assets"
TEMPLATE = BASE / "templates" / "article.html.j2"
INDEX = BASE / "index.html"

UNSPLASH_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# Expanded 30-topic interests from your history
INTERESTS = [
    "cosmic ray influence on precognition",
    "MKSEARCH psi experiments",
    "banh mi street food culture",
    "AI policy and safety",
    "cryptocurrency regulation",
    "Bitcoin market trends",
    "Comet 3I/ATLAS trajectory",
    "UFO/UAP government disclosures",
    "sacred geometry in architecture",
    "occult ritual symbolism in politics",
    "quantum computing breakthroughs",
    "esoteric numerology in media",
    "number 137 in science and culture",
    "multi-timeframe crypto trading",
    "order flow analysis in markets",
    "AI in automated journalism",
    "high-altitude particle experiments",
    "free energy research developments",
    "space exploration missions",
    "clean energy breakthroughs",
    "metaphysical event predictions",
    "geopolitical tensions in Asia",
    "off-grid living in Alaska",
    "AI art in news media",
    "predictive programming in film",
    "ancient measurement systems",
    "lunar and solar cycles",
    "stock market algorithmic trading",
    "deep sea exploration news",
    "volcanic activity and climate"
]

NUM_ARTICLES = 4

SYSTEM_PERSONA = """
Write in straight news, newspaper style, inverted pyramid:
headline → lede → key facts → context → kicker.
Neutral tone, clear sourcing, modern vocabulary.
"""

URL_FINDER_PROMPT = """
Return exactly one reputable, working news URL (no paywall if possible)
less than 6 hours old about the provided topic.
Respond with ONLY the URL.
"""

ARTICLE_PROMPT = """
Write an article in the style described in the system message, based on:
{url}

Return JSON:
- "title": headline
- "meta_description": SEO summary
- "body_html": HTML body with <p>, <h2>, <blockquote>, etc.
- "image_queries": list of 2–3 short phrases for related photos
"""

def ask_url(topic:str)->str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":URL_FINDER_PROMPT},
                  {"role":"user","content":topic}],
        temperature=0.2,
        max_tokens=200
    )
    return r.choices[0].message.content.strip()

def write_article(url:str)->dict:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":SYSTEM_PERSONA},
                  {"role":"user","content":ARTICLE_PROMPT.format(url=url)}],
        temperature=0.3,
        max_tokens=1600
    )
    text = r.choices[0].message.content.strip()
    m = re.search(r'\{.*\}\s*$', text, re.S)
    return json.loads(m.group(0) if m else text)

def unsplash(query:str)->str:
    if not UNSPLASH_KEY:
        return "assets/fallback-1.jpg"
    try:
        resp = requests.get(
            "https://api.unsplash.com/photos/random",
            params={"query":query,"orientation":"landscape","content_filter":"high"},
            headers={"Authorization":f"Client-ID {UNSPLASH_KEY}"},
            timeout=15
