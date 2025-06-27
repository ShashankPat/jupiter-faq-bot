## translate.py for the faq_bot.py for the translation in either hindi or hindi + english
from langdetect import detect
from deep_translator import GoogleTranslator
import os, openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def detect_lang(text: str) -> str:
    try:
        return "en" if detect(text) == "en" else "hi"
    except:
        return "en"

def translate(text: str, dest: str = "en") -> str:
    ## no-op if already in target
    if detect_lang(text) == dest:
        return text

    ## Try Deep Translator
    try:
        return GoogleTranslator(source="auto", target=dest).translate(text)
    except Exception:
        ## Fallback to GPT
        prompt = (
            f"Translate this into {'English' if dest=='en' else 'Hindi'}:\n\n"
            + text
        )
        try:
            resp = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            ## last-resort: give them the original
            return text
