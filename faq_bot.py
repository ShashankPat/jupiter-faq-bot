## faq_bot.py
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import openai
from translate import detect_lang, translate  ## from translate.py

## Load the OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

## Load the multilingual FAISS index & combined Q&A mapping
index = faiss.read_index("all_qa_index.faiss")
with open("all_qa_mapping.pkl", "rb") as f:
    records = pickle.load(f)

## multilingual model 
embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def get_faq_answer(
    user_query: str,
    top_k: int = 3,
    return_hits: bool = False
) -> str | tuple[str, list[dict]]:
    """
    Semantic‐search + optional GPT rephrase over the top_k FAQ hits.
    Falls back to raw top‐1 FAQ answer if LLM errors out.
    """
    ## Embed & retrieve
    q_emb = embedder.encode([user_query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    hits = [records[idx] for idx in I[0]]

    ## Build RAG prompt
    prompt = f"User asked: “{user_query}”\n\nHere are relevant FAQs:\n"
    for i, hit in enumerate(hits, 1):
        prompt += f"{i}. Q: {hit['question']}\n   A: {hit['answer']}\n"
    prompt += "\nBased on these, provide a friendly, concise answer. If unsure, say so."

    ## Try the LLM
    try:
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception:
        ## Fallback to raw retrieval
        answer = hits[0]["answer"] if hits else "Sorry, I couldn't find anything on that topic."

    if return_hits:
        return answer, hits
    return answer

def get_faq_answer_multilingual(
    user_query: str,
    top_k: int = 3,
    return_hits: bool = False
) -> str | tuple[str, list[dict]]:
    """
    Detects Hindi/Hinglish vs. English, translates IN→EN for retrieval,
    then translates the EN answer back to Hindi if needed,
    and also translates the FAQ questions when returning hits.
    """
    src = detect_lang(user_query)

    ## Translate into English if needed
    query_en = user_query if src == "en" else translate(user_query, dest="en")

    ## Retrieve & (optionally) rephrase
    if return_hits:
        answer_en, hits = get_faq_answer(query_en, top_k, return_hits=True)
    else:
        answer_en = get_faq_answer(query_en, top_k)
        hits = []

    ## Translate the answer back if needed
    answer_out = answer_en if src == "en" else translate(answer_en, dest="hi")

    ## If user asked in Hindi/Hinglish and we have hits, translate their questions too
    if src != "en" and return_hits:
        for hit in hits:
            hit["question"] = translate(hit["question"], dest="hi")

    return (answer_out, hits) if return_hits else answer_out

if __name__ == "__main__":
    while True:
        q = input("\nAsk a Jupiter question (or ‘exit’): ")
        if q.lower() in ("exit", "quit"):
            break
        print("\n" + get_faq_answer_multilingual(q) + "\n")
