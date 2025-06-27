## app.py
import streamlit as st
import numpy as np
from faq_bot import get_faq_answer_multilingual
from sentence_transformers import SentenceTransformer
from PIL import Image

st.set_page_config(page_title="Jupiter FAQ Bot", layout="wide")

## Header with logo + title centered 
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    logo = Image.open("jupiter.png")
    cols = st.columns([1, 8])
    with cols[0]:
        st.image(logo, width=50)
    with cols[1]:
        st.markdown("<h1 style='margin:0;'>Jupiter FAQ Bot</h1>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center; color:var(--secondary);'>"
        "Ask me anything about Jupiter’s features—bills, UPI, rewards, investments, and more!"
        "</p>",
        unsafe_allow_html=True
    )

embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

## Keep a little history for related‐query suggestions
if "history" not in st.session_state:
    st.session_state.history = {"queries": [], "embeddings": []}

## Callback to set the search box
def set_query(q: str):
    st.session_state.query = q

## Bind the text_input to session_state.query
query = st.text_input(
    "Your question:",
    placeholder="How do I pay my electricity bill?",
    key="query",
)

@st.cache_data(show_spinner=False)
def answered(q: str):
    ## return both the LLM answer and the raw FAQ hits
    return get_faq_answer_multilingual(q, top_k=5, return_hits=True)

if st.session_state.query:
    with st.spinner("Searching FAQs…"):
        answer, faq_hits = answered(st.session_state.query)

    left, right = st.columns([1,2])
    with left:
        st.markdown("**Your question:**")
        st.write(st.session_state.query)
    with right:
        st.markdown("**Answer:**")
        st.write(answer)

    ## log for “you might also ask…”
    emb = embedder.encode([st.session_state.query], convert_to_numpy=True)[0]
    st.session_state.history["queries"].append(st.session_state.query)
    st.session_state.history["embeddings"].append(emb)

    st.markdown("---")
    st.markdown("#### More FAQ hits…")
    for i, hit in enumerate(faq_hits[1:6], start=1):
        st.button(
            hit["question"],
            key=f"hit_{i}",
            on_click=set_query,
            kwargs={"q": hit["question"]}
        )

    ## related past queries
    if len(st.session_state.history["queries"]) > 1:
        all_embs = np.stack(st.session_state.history["embeddings"])
        cur = all_embs[-1]
        prev = all_embs[:-1]
        sims = (prev @ cur) / (np.linalg.norm(prev, axis=1) * np.linalg.norm(cur) + 1e-8)
        topk = sims.argsort()[::-1][:3]
        related = [st.session_state.history["queries"][i] for i in topk]

        with st.expander("You might also ask…"):
            for q in related:
                st.write(f"- {q}")
