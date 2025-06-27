# Jupiter FAQ Bot

A retrieval-augmented FAQ bot for Jupiter’s features (UPI, bills, rewards, investments, and more) built with FAISS, Streamlit, and OpenAI.

## Getting Started

```bash
git clone git@github.com:ShashankPat/jupiter-faq-bot.git
cd jupiter-faq-bot
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-…
streamlit run app.py


File Overview:

scrape_faqs.py → Scrapes the official Jupiter FAQ pages
scrape_forum_threads.py → Fetches recent forum threads from Discourse
clean_faqs.py → Cleans & normalizes the combined Q&A CSV
index_faqs.py → Builds a multilingual FAISS index over all Q&A
translate.py → Script for detecting/handling English ↔ Hindi
faq_bot.py → Retrieval + RAG logic (incl. bilingual support)
app.py → Streamlit front-end
evaluate_methods.py → Compares pure retrieval vs. RAG on accuracy & latency
eval_set.csv → Sample Q&A eval set for benchmarking
requirements.txt → Pinned Python deps
