# import seaborn as sns
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

vi_data_df = pd . read_csv("./vi_text_retrieval.csv")
context = vi_data_df['text']
context = [doc . lower() for doc in context]

tfidf_vectorizer = TfidfVectorizer()
context_embedded = tfidf_vectorizer.fit_transform(context)

print(context_embedded . toarray()[7][0])


def tfidf_search(question, tfidf_vectorizer, top_d=5):
    # lowercasing before encoding
    question = question.lower()
    query_embedded = tfidf_vectorizer.transform([question])
    cosine_scores = cosine_similarity(
        query_embedded, context_embedded).flatten()

    # Get top k cosine score and index its
    results = []
    for idx in cosine_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            'id': idx,
            'cosine_score': cosine_scores[idx]
        }
        results.append(doc_score)
    return results


# Example usage
question = vi_data_df.iloc[0]['question']
results = tfidf_search(question, tfidf_vectorizer, top_d=5)
print(f'results[0] {results[0]['cosine_score']}')


def corr_search(question, tfidf_vectorizer, top_d=5):
    # lowercasing before encoding
    question = question.lower()
    query_embedded = tfidf_vectorizer.transform([question])
    query_embedded = query_embedded.toarray().flatten()

    corr_scores = np.corrcoef(query_embedded, context_embedded.toarray())
    corr_scores = corr_scores[0][1:]
    # Get top k correlation score and index its
    results = []
    for idx in corr_scores . argsort()[- top_d:][:: -1]:
        doc = {
            'id': idx,
            'corr_score': corr_scores[idx]
        }
        results . append(doc)
    return results


question = vi_data_df . iloc[0]['question']
results = corr_search(question, tfidf_vectorizer, top_d=5)
print(f'corr score: {results[1]['corr_score']}')
