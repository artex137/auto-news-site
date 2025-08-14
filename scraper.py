# scraper.py
import asyncio
from typing import Dict, List, Tuple
from urllib.parse import urlparse
import re

from bs4 import BeautifulSoup  # installed in requirements
from playwright.async_api import async_playwright

def _extract_from_html(html: str) -> Tuple[str, List[str]]:
    soup = BeautifulSoup(html, "lxml")

    # Prefer <article>, then best <main>/<section>/<div> by paragraph count
    def best_text_container():
        art = soup.find("article")
        if art:
            return art
        candidates = []
        for sel in ["main", "section", "div"]:
            for el in soup.find_all(sel):
                pcount = len(el.find_all("p"))
                cls = " ".join(el.get("class", []))
                score = pcount
                if any(k in (cls or "").lower() for k in ["content","article","story","post","entry","body","read"]):
                    score += 4
                if score >= 6:
                    candidates.append((score, el))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        return soup

    container = best_text_container()
    text = " ".join(p.get_text(" ", strip=True) for p in container.find_all(["p","li"]))
    text = re.sub(r"\s+", " ", text or "").strip()

    # Images: og:image + images inside container
    images = set()
    for m in soup.find_all("meta", attrs={"property":"og:image"}):
        if m.get("content"): images.add(m["content"])
    for m in soup.find_all("meta", attrs={"name":"twitter:image"}):
        if m.get("content"): images.add(m["content"])
    for im in container.find_all("img", src=True):
        src = im["src"]
        if src.startswith("data:"): continue
        images.add(src)

    # Prefer larger-looking URLs
    ordered = sorted(images, key=lambda u: (("=w" in u or "width" in u or "2048" in u or "1600" in u), len(u)), reverse=True)
    return text, ordered[:6]

async def _scrape(url: str, timeout_ms: int = 20000) -> Tuple[str, List[str]]:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36")
        page = await context.new_page()
        try:
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            # Many news sites lazy-load—give a moment, then try networkidle
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            html = await page.content()
            text, images = _extract_from_html(html)
            # If text too short, try scrolling to trigger lazy content
            if len(text) < 300:
                for _ in range(4):
                    await page.mouse.wheel(0, 2000)
                    await page.wait_for_timeout(800)
                html = await page.content()
                text, images = _extract_from_html(html)
            return text, images
        finally:
            await context.close()
            await browser.close()

def scrape_article(url: str) -> Tuple[str, List[str]]:
    """Sync wrapper for async Playwright scraper."""
    try:
        return asyncio.run(_scrape(url))
    except RuntimeError:
        # If already in an event loop (rare in GH Actions), create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_scrape(url))
        finally:
            loop.close()
