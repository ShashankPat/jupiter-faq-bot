# scrape_faqs.py
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def fetch_page(url: str) -> BeautifulSoup:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page    = browser.new_page()
        page.goto(url)
        html    = page.content()
        browser.close()
    return BeautifulSoup(html, "html.parser")

def parse_faqs(soup: BeautifulSoup, category: str) -> list[dict]:
    text  = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    start = next(
        (i for i, l in enumerate(lines)
        if "frequently asked questions" in l.lower()),
        None
    )
    if start is None:
        return []

    faqs = []
    q = None
    for line in lines[start+1:]:
        low = line.lower()
        if any(stop in low for stop in ("related posts", "share", "comments are closed")):
            break
        if line.endswith("?"):
            q = line
        elif q:
            faqs.append({
                "category": category,
                "question": q,
                "answer":   line
            })
            q = None
    return faqs

def scrape_all(base_url: str, categories: list[str]) -> pd.DataFrame:
    rows = []
    for cat in categories:
        url   = f"{base_url}/{cat}"
        soup  = fetch_page(url)
        batch = parse_faqs(soup, cat)
        print(f" • {cat}: scraped {len(batch)} items")
        rows.extend(batch)
    return pd.DataFrame(rows)

## Crawl the 4 official Jupiter FAQ pages and return a DataFrame of {category, question, answer}
def scrape_official_faqs() -> pd.DataFrame:
    base_url = "https://jupiter.money"
    categories = [
        "pay-via-upi",
        "bills-recharges",
        "rewards",
        "investments",
    ]
    return scrape_all(base_url, categories)

if __name__ == "__main__":
    df = scrape_official_faqs()
    df.to_csv("jupiter_faqs_raw.csv", index=False)
    print(f"\nTotal FAQs scraped: {len(df)}")
    print("Saved to jupiter_faqs_raw.csv")
