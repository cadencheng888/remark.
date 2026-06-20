import os
import json
import time
from dotenv import load_dotenv
from browserbase import Browserbase
from playwright.sync_api import sync_playwright

load_dotenv()

API_KEY = os.environ["BROWSERBASE_API_KEY"]
PROJECT_ID = os.environ["BROWSERBASE_PROJECT_ID"]

bb = Browserbase(api_key=API_KEY)

SEARCH_QUERY = "baggy washed barrel jean"


def scrape_google_shopping(page) -> list[dict]:
    url = f"https://www.google.com/search?q={SEARCH_QUERY.replace(' ', '+')}&tbm=shop"
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    time.sleep(2)

    results = page.evaluate("""
        () => {
            const items = [];
            document.querySelectorAll('.sh-dgr__grid-result, .u30d4').forEach(el => {
                const title = el.querySelector('h3, .tAxDx')?.innerText?.trim();
                const price = el.querySelector('.a8Pemb, .HRLxBb')?.innerText?.trim();
                const link = el.querySelector('a')?.href;
                const store = el.querySelector('.aULzUe, .IuHnof')?.innerText?.trim();
                if (title && price) items.push({ title, price, store: store || 'Google Shopping', link, source: 'Google Shopping' });
            });
            return items;
        }
    """)
    return results


def scrape_asos(page) -> list[dict]:
    url = f"https://www.asos.com/us/search/?q={SEARCH_QUERY.replace(' ', '+')}"
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    time.sleep(3)

    results = page.evaluate("""
        () => {
            const items = [];
            document.querySelectorAll('article[data-auto-id="productTile"]').forEach(el => {
                const title = el.querySelector('[data-auto-id="productTileDescription"]')?.innerText?.trim();
                const price = el.querySelector('[data-auto-id="productTilePrice"]')?.innerText?.trim();
                const link = el.querySelector('a')?.href;
                if (title && price) items.push({ title, price, store: 'ASOS', link, source: 'ASOS' });
            });
            return items;
        }
    """)
    return results


def scrape_amazon(page) -> list[dict]:
    url = f"https://www.amazon.com/s?k={SEARCH_QUERY.replace(' ', '+')}"
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    time.sleep(2)

    results = page.evaluate("""
        () => {
            const items = [];
            document.querySelectorAll('[data-component-type="s-search-result"]').forEach(el => {
                const title = el.querySelector('h2 span')?.innerText?.trim();
                const whole = el.querySelector('.a-price-whole')?.innerText?.trim();
                const fraction = el.querySelector('.a-price-fraction')?.innerText?.trim();
                const price = whole ? `$${whole}${fraction ? '.' + fraction : ''}` : null;
                const link = el.querySelector('h2 a')?.href;
                if (title && price) items.push({ title, price, store: 'Amazon', link, source: 'Amazon' });
            });
            return items;
        }
    """)
    return results


def parse_price(price_str: str) -> float:
    """Extract a float from a price string like '$34.99' or '$34 $29'."""
    import re
    # grab the last price if there are multiple (usually the sale price)
    matches = re.findall(r'\$?([\d,]+\.?\d*)', price_str.replace(',', ''))
    if not matches:
        return float('inf')
    return float(matches[-1])


def run():
    session = bb.sessions.create(project_id=PROJECT_ID)
    all_results = []

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(session.connect_url)
        context = browser.contexts[0]

        scrapers = [
            ("Google Shopping", scrape_google_shopping),
            ("ASOS", scrape_asos),
            ("Amazon", scrape_amazon),
        ]

        for name, scraper in scrapers:
            print(f"Scraping {name}...")
            try:
                page = context.new_page()
                results = scraper(page)
                print(f"  Found {len(results)} results")
                all_results.extend(results)
                page.close()
            except Exception as e:
                print(f"  {name} failed: {e}")

        browser.close()

    # sort by price ascending
    all_results.sort(key=lambda x: parse_price(x.get("price", "")))

    print(f"\n{'='*60}")
    print(f"TOP DEALS FOR: {SEARCH_QUERY.upper()}")
    print(f"{'='*60}")
    for i, item in enumerate(all_results[:20], 1):
        print(f"\n#{i} [{item['source']}]")
        print(f"  {item['title']}")
        print(f"  Price: {item['price']}")
        if item.get('store') and item['store'] != item['source']:
            print(f"  Store: {item['store']}")
        if item.get('link'):
            print(f"  Link:  {item['link'][:80]}...")

    with open("deals_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to deals_results.json")


if __name__ == "__main__":
    run()
