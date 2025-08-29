import numpy as np
import faiss
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import os
import gc
import torch

class RetrievalSystem:
    def __init__(self, method="dense", model_path=None, device=None):
        """
        Create a retrieval system

        Args:
        method: retrieve type ("dense", "bm25", "tfidf", "hybrid")
        model_path: Embedding model path
        device: Computing device (GPU/CPU)
        """
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if method in ["dense", "hybrid"] and model_path:
            print(f"Loading embedding model: {model_path}")
            self.embedding_model = SentenceTransformer(model_path, device=self.device)
        else:
            self.embedding_model = None
        
        self.chunks = None
        self.faiss_index = None
        self.bm25_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.is_ready = False
        
    def load_chunks_and_index(self, chunk_file: str, faiss_index_file: str = None):
        """Load chunks and index"""
        print(f"Loading chunks from {chunk_file}...")
        
        # load chunks
        try:
            df = pd.read_csv(chunk_file, encoding='utf-8')
            self.chunks = df.to_dict('records')
            print(f"✓ Loaded {len(self.chunks)} chunks")
        except Exception as e:
            print(f"Error loading chunks: {e}")
            return False
        
        # load FAISS index for dense method
        if self.method in ["dense", "hybrid"] and faiss_index_file and os.path.exists(faiss_index_file):
            try:
                print(f"Loading FAISS index from {faiss_index_file}...")
                self.faiss_index = faiss.read_index(faiss_index_file)
                print(f"✓ Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                return False
        
        # create BM25 index for BM25
        if self.method in ["bm25", "hybrid"]:
            print("Building BM25 index...")
            try:
                chunk_texts = [chunk['text'] for chunk in self.chunks]
                tokenized_chunks = [text.split() for text in chunk_texts]
                self.bm25_index = BM25Okapi(tokenized_chunks)
                print("✓ BM25 index built successfully")
            except Exception as e:
                print(f"Error building BM25 index: {e}")
                return False
        
        # create TF-IDF index
        if self.method in ["tfidf", "hybrid"]:
            print("Building TF-IDF index...")
            try:
                chunk_texts = [chunk['text'] for chunk in self.chunks]
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=10000,
                    stop_words=None,  # non for Persian
                    ngram_range=(1, 2)
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
                print("✓ TF-IDF index built successfully")
            except Exception as e:
                print(f"Error building TF-IDF index: {e}")
                return False
        
        self.is_ready = True
        return True
    
    def retrieve_dense(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Retrieve dense (semantic)"""
        if not self.embedding_model or not self.faiss_index:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query], device=self.device)
            query_embedding = query_embedding.astype('float32')
            
            # search in FAISS
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(self.chunks): 
                    chunk = self.chunks[idx]
                    similarity = 1 / (1 + distance)  # for L2 distance
                    results.append((chunk, similarity))
            
            return results
            
        except Exception as e:
            print(f"Error in dense retrieval: {e}")
            return []
    
    def retrieve_bm25(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Retrieve BM25"""
        if not self.bm25_index:
            return []
        
        try:
            # tokenize query
            query_tokens = query.split()
            
            # محاسبه امتیازات BM25
            scores = self.bm25_index.get_scores(query_tokens)
            
            # مرتب‌سازی و انتخاب top_k
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    score = scores[idx]
                    results.append((chunk, score))
            
            return results
            
        except Exception as e:
            print(f"Error in BM25 retrieval: {e}")
            return []
    
    def retrieve_tfidf(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Retrieve TF-IDF"""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return []
        
        try:
            # تبدیل query به TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # محاسبه cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # مرتب‌سازی و انتخاب top_k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    score = similarities[idx]
                    results.append((chunk, score))
            
            return results
            
        except Exception as e:
            print(f"Error in TF-IDF retrieval: {e}")
            return []
    
    def retrieve_hybrid(self, query: str, top_k: int = 10, 
                       dense_weight: float = 0.6, bm25_weight: float = 0.4) -> List[Tuple[Dict, float]]:
        """Retrieve hybrid (Dense + BM25)"""
        try:
            dense_results = self.retrieve_dense(query, top_k * 2)
            bm25_results = self.retrieve_bm25(query, top_k * 2)
            
            combined_scores = {}
            
            if dense_results:
                max_dense_score = max([score for _, score in dense_results])
                for chunk, score in dense_results:
                    chunk_id = chunk['id']
                    normalized_score = score / max_dense_score if max_dense_score > 0 else 0
                    combined_scores[chunk_id] = {
                        'chunk': chunk,
                        'dense_score': normalized_score * dense_weight,
                        'bm25_score': 0
                    }
            
            if bm25_results:
                max_bm25_score = max([score for _, score in bm25_results])
                for chunk, score in bm25_results:
                    chunk_id = chunk['id']
                    normalized_score = score / max_bm25_score if max_bm25_score > 0 else 0
                    
                    if chunk_id in combined_scores:
                        combined_scores[chunk_id]['bm25_score'] = normalized_score * bm25_weight
                    else:
                        combined_scores[chunk_id] = {
                            'chunk': chunk,
                            'dense_score': 0,
                            'bm25_score': normalized_score * bm25_weight
                        }
            
            final_results = []
            for chunk_id, data in combined_scores.items():
                final_score = data['dense_score'] + data['bm25_score']
                final_results.append((data['chunk'], final_score))
            
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            return final_results[:top_k]
            
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")
            return []
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Main method for retrieve based on the selected method"""
        if not self.is_ready:
            print("Retrieval system is not ready. Please load chunks and index first.")
            return []
        
        if self.method == "dense":
            return self.retrieve_dense(query, top_k)
        elif self.method == "bm25":
            return self.retrieve_bm25(query, top_k)
        elif self.method == "tfidf":
            return self.retrieve_tfidf(query, top_k)
        elif self.method == "hybrid":
            return self.retrieve_hybrid(query, top_k)
        else:
            print(f"Unknown retrieval method: {self.method}")
            return []
    
    def get_contexts_for_rag(self, query: str, top_k: int = 5, 
                            max_context_length: int = 2000) -> Tuple[List[str], List[Dict]]:
        """get contexts for RAG"""
        retrieved_results = self.retrieve(query, top_k)
        
        contexts = []
        metadata = []
        total_length = 0
        
        for chunk, score in retrieved_results:
            chunk_text = chunk['text']
            
            if total_length + len(chunk_text) > max_context_length:
                remaining_length = max_context_length - total_length
                if remaining_length > 100:
                    chunk_text = chunk_text[:remaining_length] + "..."
                else:
                    break
            
            contexts.append(chunk_text)
            metadata.append({
                'chunk_id': chunk['id'],
                'score': score,
                'chunk_type': chunk.get('chunk_type', 'unknown'),
                'length': len(chunk_text)
            })
            
            total_length += len(chunk_text)
            
            if total_length >= max_context_length:
                break
        
        return contexts, metadata
    
    def evaluate_retrieval_quality(self, test_queries: List[Dict], 
                                 relevant_chunks: Dict[str, List[str]]) -> Dict[str, float]:
        """Evaluation of Retrieval Quality"""
        print(f"Evaluating retrieval quality on {len(test_queries)} queries...")
        
        hit_at_1 = []
        hit_at_3 = []
        hit_at_5 = []
        mrr_scores = []
        
        for i, query_data in enumerate(test_queries):
            if i % 50 == 0:
                print(f"  Processing query {i+1}/{len(test_queries)}")
            
            query = query_data['question']
            query_id = query_data.get('id', str(i))
            
            relevant = relevant_chunks.get(query_id, [])
            if not relevant:
                continue
            
            retrieved_results = self.retrieve(query, top_k=10)
            retrieved_ids = [chunk['id'] for chunk, _ in retrieved_results]
            
            hit_at_1.append(any(chunk_id in relevant for chunk_id in retrieved_ids[:1]))
            hit_at_3.append(any(chunk_id in relevant for chunk_id in retrieved_ids[:3]))
            hit_at_5.append(any(chunk_id in relevant for chunk_id in retrieved_ids[:5]))
            
            mrr = 0.0
            for rank, chunk_id in enumerate(retrieved_ids, 1):
                if chunk_id in relevant:
                    mrr = 1.0 / rank
                    break
            mrr_scores.append(mrr)
        
        results = {
            'hit_at_1': np.mean(hit_at_1) if hit_at_1 else 0.0,
            'hit_at_3': np.mean(hit_at_3) if hit_at_3 else 0.0,
            'hit_at_5': np.mean(hit_at_5) if hit_at_5 else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'total_queries': len(test_queries)
        }
        
        print("✓ Retrieval evaluation completed")
        print(f"  Hit@1: {results['hit_at_1']:.3f}")
        print(f"  Hit@3: {results['hit_at_3']:.3f}")
        print(f"  Hit@5: {results['hit_at_5']:.3f}")
        print(f"  MRR: {results['mrr']:.3f}")
        
        return results
    
    def cleanup(self):
        """Clean up some space"""
        if hasattr(self, 'embedding_model'):
            del self.embedding_model
        if hasattr(self, 'faiss_index'):
            del self.faiss_index
        if hasattr(self, 'chunks'):
            del self.chunks
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MultiModelRetrieval:
    """Class for comparing multiple embedding models in retrieval"""
    
    def __init__(self, model_paths: List[str], device=None):
        self.model_paths = model_paths
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.retrievers = {}
        
    def setup_retrievers(self, chunk_file: str, faiss_indices: Dict[str, str]):
        """Setting up retrievers for all models"""
        print("Setting up retrievers for all models...")
        
        for model_path in self.model_paths:
            model_name = os.path.basename(model_path)
            print(f"\n--- Setting up retriever for {model_name} ---")
            
            try:
                retriever = RetrievalSystem(
                    method="dense",
                    model_path=model_path,
                    device=self.device
                )
                
                faiss_index_file = faiss_indices.get(model_name)
                if retriever.load_chunks_and_index(chunk_file, faiss_index_file):
                    self.retrievers[model_name] = retriever
                    print(f"✓ {model_name} retriever ready")
                else:
                    print(f"✗ Failed to setup {model_name} retriever")
                    
            except Exception as e:
                print(f"Error setting up {model_name}: {e}")
    
    def compare_retrieval_performance(self, test_queries: List[Dict], 
                                    relevant_chunks: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Compare the performance of all models"""
        results = {}
        
        for model_name, retriever in self.retrievers.items():
            print(f"\n=== Evaluating {model_name} ===")
            model_results = retriever.evaluate_retrieval_quality(test_queries, relevant_chunks)
            results[model_name] = model_results
        
        return results
    
    def cleanup_all(self):
        """Clean up retrievers"""
        for retriever in self.retrievers.values():
            retriever.cleanup()
        self.retrievers.clear()
        gc.collect()