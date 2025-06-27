# clean_all.py
import pandas as pd
import html
import re

## raw combined Q&A
df = pd.read_csv("jupiter_all_qa.csv")

## Pre-cleaning removals before lowercasing/unescaping:
## Drop any “img_<whatever>” tokens (e.g. img_5261)
df["answer"] = df["answer"].str.replace(
    r"\bimg_[A-Za-z0-9_.-]+\b",
    "",
    regex=True,
)

## Remove file-size artifacts, e.g. “11702532 67.9 kb” or “12345 1.2 MB”
df["answer"] = df["answer"].str.replace(
    r"\b\d+\s+\d+(\.\d+)?\s*(?:bytes|kb|mb)\b",
    "",
    regex=True,
)

## Strip any lingering HTML tags
for col in ("question", "answer"):
    df[col] = df[col].str.replace(r"<[^>]+>", "", regex=True)

## Remove emojis / non-ASCII symbols
for col in ("question", "answer"):
    df[col] = df[col].str.replace(r"[^\x00-\x7F]+", "", regex=True)

## Collapse multiple blank lines in answers
df["answer"] = df["answer"].str.replace(r"\n\s*\n+", "\n", regex=True)

## Normalize text: strip, lowercase, unescape HTML entities
for col in ("question", "answer"):
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(html.unescape)
    )

## Drop exact duplicates (same question + same answer)
before = len(df)
df = df.drop_duplicates(subset=["question", "answer"])
print(f"Dropped {before - len(df)} exact duplicates")

## Remove any stray “questions?” rows
df = df[~df["question"].str.fullmatch(r"questions\?")]

## Drop extremely short questions (< 5 chars)
df = df[df["question"].str.len() > 5]

df.to_csv("jupiter_all_qa_clean.csv", index=False)
print(f"Cleaned Q&A: {len(df)} rows → jupiter_all_qa_clean.csv")
