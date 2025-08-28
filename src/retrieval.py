from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np

class RetrievalSystem:
    def __init__(self, method="bm25"):
        self.method = method
        self.index = None
    
    def build_index(self, chunks, embeddings=None):
        """ساخت ایندکس برای بازیابی"""
        if self.method == "bm25":
            self.index = BM25Okapi([chunk.split() for chunk in chunks])
        elif self.method == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.index = self.vectorizer.fit_transform(chunks)
        elif self.method == "dense":
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings.astype('float32'))
    
    def retrieve(self, query, k=5):
        """بازیابی متون مرتبط"""
        pass