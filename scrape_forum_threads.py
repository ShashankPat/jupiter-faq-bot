# scrape_forum_threads.py
import pandas as pd
import requests
from bs4 import BeautifulSoup

## Fetch the latest `limit` topics in this Discourse category via its JSON API and returns a list of dicts with keys: 
## category, question (topic title), answer (first post text).
def fetch_forum_topics(category_slug: str, limit: int = 10) -> list[dict]:
    url = f"https://community.jupiter.money/c/{category_slug}.json"
    resp = requests.get(url)
    if not resp.ok:
        return []

    topics = resp.json().get("topic_list", {}).get("topics", [])[:limit]
    rows = []
    for topic in topics:
        tid   = topic.get("id")
        title = topic.get("title", "")

        pj = requests.get(f"https://community.jupiter.money/t/{tid}/posts.json")
        if not pj.ok:
            continue
        posts = pj.json().get("post_stream", {}).get("posts", [])
        if not posts:
            continue

        cooked = posts[0].get("cooked", "")
        answer = BeautifulSoup(cooked, "html.parser").get_text("\n", strip=True)

        rows.append({
            "category": category_slug,
            "question": title,
            "answer":   answer
        })
    return rows

## Iterate over all forum categories, fetch up to `limit_per_cat` topics per category,and return a DataFrame of {category, question, answer}.
def scrape_forum_threads(limit_per_cat: int = 10) -> pd.DataFrame:
    
    forum_categories = [
        "feedback-ideas","news-from-jupiter","jupiter-labs","investments-club",
        "bug-hunters","my-finance","hall-of-fame","research-centre",
        "general","help","all-things-else",
    ]

    all_topics = []
    for slug in forum_categories:
        topics = fetch_forum_topics(slug, limit=limit_per_cat)
        print(f" • {slug}: fetched {len(topics)} threads")
        all_topics.extend(topics)

    return pd.DataFrame(all_topics)

if __name__ == "__main__":
    df = scrape_forum_threads(limit_per_cat=10)
    df.to_csv("jupiter_forum_threads.csv", index=False)
    print(f"Saved to jupiter_forum_threads.csv ({len(df)} threads)") 
