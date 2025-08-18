# scraper.py
import asyncio
from typing import List, Tuple
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

MIN_IMG_WIDTH  = 800
MIN_IMG_HEIGHT = 450
MIN_AR = 1.2   # prefer landscape >= 1.2
MAX_AR = 3.2

def _extract_fallback_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    art = soup.find("article")
    container = art or soup
    text = " ".join(p.get_text(" ", strip=True) for p in container.find_all(["p","li"]))
    return " ".join(text.split())

async def _scrape(url: str, timeout_ms: int = 25000) -> Tuple[str, List[str]]:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
        page = await context.new_page()
        try:
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass

            # Prefer a real <article> text if present
            html = await page.content()
            soup = BeautifulSoup(html, "lxml")
            article = soup.find("article")
            if article:
                text = " ".join(p.get_text(" ", strip=True) for p in article.find_all(["p","li"]))
            else:
                text = ""
            if len(text) < 300:
                text = _extract_fallback_text(html)

            # Collect high-quality images using the rendered DOM (natural size)
            js = """
            () => {
              function scoreImg(img) {
                const w = img.naturalWidth || 0;
                const h = img.naturalHeight || 0;
                if (w < 1 || h < 1) return -1;
                const ar = w / h;
                let s = 0;
                if (w >= %d && h >= %d) s += 5;
                if (ar >= %f && ar <= %f) s += 3; // landscape preference
                // deweight tiny thumbs/icons/logos
                if (w < 64 || h < 64) s -= 5;
                const src = (img.currentSrc || img.src || "").toLowerCase();
                // penalize obvious ui images
                const bad = ["sprite","icon","logo","placeholder","thumb","avatar","tracking","pixel","adsystem"];
                if (bad.some(b => src.includes(b))) s -= 4;
                return s + Math.log(1 + w*h)/10.0;
              }

              const scope = document.querySelector("article") || document.body;
              const imgs = Array.from(scope.querySelectorAll("img[src]"));
              const candidates = imgs
                 .map(img => ({src: img.currentSrc || img.src, w: img.naturalWidth||0, h: img.naturalHeight||0, s: scoreImg(img)}))
                 .filter(o => o.s > 0)
                 .sort((a,b) => b.s - a.s);
              return candidates.slice(0, 8);
            }
            """ % (MIN_IMG_WIDTH, MIN_IMG_HEIGHT, MIN_AR, MAX_AR)
            cands = await page.evaluate(js)
            abs_urls: List[str] = []
            base = page.url
            for c in cands:
                u = c.get("src") or ""
                if not u or u.startswith("data:"): continue
                abs_urls.append(urljoin(base, u))

            return " ".join(text.split()), abs_urls
        finally:
            await context.close()
            await browser.close()

def scrape_article(url: str) -> Tuple[str, List[str]]:
    try:
        return asyncio.run(_scrape(url))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_scrape(url))
        finally:
            loop.close()
