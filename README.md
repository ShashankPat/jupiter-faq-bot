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
