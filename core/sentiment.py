from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

def doc_sentiment(text: str):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text or "")

def rolling_sentiment(text: str, window_sentences: int = 3):
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    sia = SentimentIntensityAnalyzer()
    rows = []
    for i in range(0, len(sents), window_sentences):
        chunk = " ".join(sents[i:i+window_sentences])
        if not chunk:
            continue
        rows.append({
            "index": i // window_sentences,
            "compound": sia.polarity_scores(chunk)["compound"]
        })
    return rows
