## test_retrival.py for checking before implementing the faq_bot
import faiss, pickle
from sentence_transformers import SentenceTransformer

## Load the combined FAISS index & mapping
index = faiss.read_index("all_qa_index.faiss")
with open("all_qa_mapping.pkl", "rb") as f:
    records = pickle.load(f)

## Load the multilingual embedder
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def retrieve(query, top_k=3):
    emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(emb, top_k)
    for dist, idx in zip(D[0], I[0]):
        r = records[idx]
        print(f"\nQ: {r['question']}\nA: {r['answer']}\n(score: {dist:.2f})")

if __name__ == "__main__":
    sample_queries = [
        "how do i pay my electricity bill?",
        "what rewards can i redeem?",
        "can i open a savings account on Jupiter?",
        "is there a limit on transactions?",
        "how does investments club work?"
    ]
    for q in sample_queries:
        print(f"\n=== Query: {q}")
        retrieve(q)
