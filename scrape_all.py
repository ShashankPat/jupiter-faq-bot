# scrape_all.py
import pandas as pd
from scrape_faqs import scrape_official_faqs
from scrape_forum_threads import scrape_forum_threads

def main():
    ## official FAQ scraper
    df_faqs = scrape_official_faqs()
    df_faqs["source"] = "official"

    ## forum-threads scraper (10 topics per category)
    df_forum = scrape_forum_threads(limit_per_cat=10)
    df_forum["source"] = "forum"

    ## Concatenate
    df_all = pd.concat([df_faqs, df_forum], ignore_index=True)

    df_all.to_csv("jupiter_all_qa.csv", index=False)
    print(f"Total Q&A pairs: {len(df_all)} → jupiter_all_qa.csv")

if __name__ == "__main__":
    main()
