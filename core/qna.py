from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def split_sentences(text: str):
    return [s.strip() for s in sent_tokenize(text) if s.strip()]

def answer_question(question: str, sentences, top_k: int = 3):
    if not sentences:
        return []
    vect = TfidfVectorizer().fit(sentences + [question])
    S = vect.transform(sentences)
    q = vect.transform([question])
    sims = cosine_similarity(q, S).ravel()
    idxs = sims.argsort()[::-1][:top_k]
    return [(float(sims[i]), sentences[i]) for i in idxs]
