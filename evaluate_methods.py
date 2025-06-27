# evaluate_methods.py
import time
import pandas as pd
from rapidfuzz import fuzz
from faq_bot import get_faq_answer, get_faq_answer_multilingual  ## from the faq_bot

## Pure retrieval: take the top‐k FAQ answer without GPT rewriting.
def retrieval_only(query: str, top_k: int) -> str:
    answer, _ = get_faq_answer_multilingual(query, top_k=top_k, return_hits=True)
    return answer

## RAG: retrieval + GPT rewrite over top‐k FAQs.
def llm_augmented(query: str, top_k: int) -> str:
    return get_faq_answer(query, top_k=top_k)

## Compute a fuzzy‐match accuracy score (0–100) against the gold answer.
def score_accuracy(pred: str, gold: str) -> float:
    return fuzz.token_sort_ratio(pred, gold)

## Run fn(*args, **kwargs), return (output, elapsed_seconds)
def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0

def main():
    df = pd.read_csv("eval_set.csv")
    top_k_values = [1, 3, 5]
    records = []

    for top_k in top_k_values:
        for _, row in df.iterrows():
            q, gold = row["query"], row["gold_answer"]

            #3 Retrieval‐only
            pred_ret, t_ret = time_call(retrieval_only, q, top_k=top_k)
            acc_ret = score_accuracy(pred_ret, gold)

            ## LLM‐augmented
            pred_llm, t_llm = time_call(llm_augmented, q, top_k=top_k)
            acc_llm = score_accuracy(pred_llm, gold)

            records.append({
                "method": "retrieval_only",
                "top_k": top_k,
                "query": q,
                "gold": gold,
                "prediction": pred_ret,
                "time_s": t_ret,
                "accuracy": acc_ret
            })
            records.append({
                "method": "llm_augmented",
                "top_k": top_k,
                "query": q,
                "gold": gold,
                "prediction": pred_llm,
                "time_s": t_llm,
                "accuracy": acc_llm
            })

    results = pd.DataFrame(records)
    results.to_csv("evaluation_results.csv", index=False)

    ## Print mean accuracy & latency by method and top_k
    summary = results.groupby(["method","top_k"])[["accuracy","time_s"]].mean()
    print(summary)

if __name__ == "__main__":
    main()
