import os
from dotenv import load_dotenv
from browserbase import Browserbase
from playwright.sync_api import sync_playwright

load_dotenv()

API_KEY = os.environ["BROWSERBASE_API_KEY"]
PROJECT_ID = os.environ["BROWSERBASE_PROJECT_ID"]

bb = Browserbase(api_key=API_KEY)

url_link = "https://www.pacsun.com/?srsltid=AfmBOoqKRoPhwv26MZaKUZ-aQb9elpS7bnWCwcARnaPV0DCX8Fa_towX"

def test_basic_navigation():
    """Test that a session can navigate to a URL and return the page title."""
    session = bb.sessions.create(project_id=PROJECT_ID)
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(session.connect_url)
        page = browser.contexts[0].pages[0]
        page.goto(url_link)
        title = page.title()
        print(f"Page title: {title}")
        assert "Example" in title, f"Unexpected title: {title}"
        browser.close()
    print("test_basic_navigation passed")


def test_screenshot():
    """Test that a screenshot can be taken in a Browserbase session."""
    session = bb.sessions.create(project_id=PROJECT_ID)
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(session.connect_url)
        page = browser.contexts[0].pages[0]
        page.goto(url_link)
        screenshot = page.screenshot()
        assert len(screenshot) > 0, "Screenshot is empty"
        print(f"Screenshot size: {len(screenshot)} bytes")
        browser.close()
    print("test_screenshot passed")


def test_page_content():
    """Test that page content can be extracted."""
    session = bb.sessions.create(project_id=PROJECT_ID)
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(session.connect_url)
        page = browser.contexts[0].pages[0]
        page.goto(url_link)
        content = page.content()
        assert "<html" in content.lower(), "No HTML content found"
        print(f"Page content length: {len(content)} chars")
        browser.close()
    print("test_page_content passed")


if __name__ == "__main__":
    tests = [test_basic_navigation, test_screenshot, test_page_content]
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"{test.__name__} FAILED: {e}")
