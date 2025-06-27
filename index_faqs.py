# index_faqs.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def build_index(
    clean_csv="jupiter_all_qa_clean.csv",   ## cleaned CSV
    model_name="paraphrase-multilingual-mpnet-base-v2",
    index_path="all_qa_index.faiss",
    mapping_path="all_qa_mapping.pkl"
):
    ## Load the cleaned, deduped Q&A
    df = pd.read_csv(clean_csv)

    ## Drop extremely short questions (< 6 chars)
    df = df[df["question"].str.len() > 5].reset_index(drop=True)

    ## Extract the question strings
    questions = df["question"].tolist()

    ## Compute embeddings with your multilingual model
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        questions,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    ## Build a FAISS L2 index over the embeddings
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index = faiss.IndexIDMap(index)
    ids = np.arange(len(questions))
    index.add_with_ids(embeddings, ids)

    ## Persist both the index and the mapping (full records)
    faiss.write_index(index, index_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    print(f"Built FAISS index with {len(questions)} vectors")

if __name__ == "__main__":
    build_index()
