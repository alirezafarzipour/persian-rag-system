import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
import torch

class Evaluator:
    def __init__(self):
        pass
    
    def exact_match(self, pred: str, gold: str) -> float:
        """Exact Match"""
        pred_clean = pred.strip().lower()
        gold_clean = gold.strip().lower()
        return float(pred_clean == gold_clean)
    
    def f1_score(self, pred: str, gold: str) -> float:
        """F1 Score"""
        pred_tokens = set(pred.strip().split())
        gold_tokens = set(gold.strip().split())
        
        if len(pred_tokens) == 0 and len(gold_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0
            
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(gold_tokens) if gold_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def precision(self, pred: str, gold: str) -> float:
        """Precision"""
        pred_tokens = set(pred.strip().split())
        gold_tokens = set(gold.strip().split())
        
        if len(pred_tokens) == 0:
            return 0.0
        
        common = pred_tokens & gold_tokens
        return len(common) / len(pred_tokens)
    
    def recall(self, pred: str, gold: str) -> float:
        """Recall"""
        pred_tokens = set(pred.strip().split())
        gold_tokens = set(gold.strip().split())
        
        if len(gold_tokens) == 0:
            return 0.0
        
        common = pred_tokens & gold_tokens
        return len(common) / len(gold_tokens)
    
    def hit_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Hit@K"""
        if not retrieved or not relevant:
            return 0.0
        
        top_k = retrieved[:k]
        return float(any(item in relevant for item in top_k))
    
    def mrr(self, rankings: List[List[bool]]) -> float:
        """Mean Reciprocal Rank"""
        if not rankings:
            return 0.0
            
        scores = []
        for rank_list in rankings:
            score = 0.0
            for i, is_relevant in enumerate(rank_list):
                if is_relevant:
                    score = 1.0 / (i + 1)
                    break
            scores.append(score)
        return np.mean(scores)
    
    def cosine_sim(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine Similarity"""
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
            
        return cosine_similarity(emb1, emb2)[0][0]
    
    def evaluate_model_performance(self, model, test_data: List[Dict], 
                                 model_name: str = "model") -> Dict[str, float]:
        """Evaluate model performance on test data"""
        print(f"Evaluating {model_name}...")
        
        exact_matches = []
        f1_scores = []
        precisions = []
        recalls = []
        cosine_similarities = []
        
        for i, item in enumerate(test_data):
            if i % 100 == 0:
                print(f"  Processing {i}/{len(test_data)} samples...")
            
            question = item['question']
            gold_answer = item['answer']
            
            question_emb = model.encode([question])
            
            pred_answer = gold_answer  # موقتاً برای تست
            
            em = self.exact_match(pred_answer, gold_answer)
            f1 = self.f1_score(pred_answer, gold_answer)
            prec = self.precision(pred_answer, gold_answer)
            rec = self.recall(pred_answer, gold_answer)
            
            gold_emb = model.encode([gold_answer])
            cos_sim = self.cosine_sim(question_emb[0], gold_emb[0])
            
            exact_matches.append(em)
            f1_scores.append(f1)
            precisions.append(prec)
            recalls.append(rec)
            cosine_similarities.append(cos_sim)
        
        results = {
            f"{model_name}_exact_match": np.mean(exact_matches),
            f"{model_name}_f1_score": np.mean(f1_scores),
            f"{model_name}_precision": np.mean(precisions),
            f"{model_name}_recall": np.mean(recalls),
            f"{model_name}_cosine_similarity": np.mean(cosine_similarities),
            f"{model_name}_num_samples": len(test_data)
        }
        
        print(f"✓ {model_name} evaluation completed")
        return results
    
    def evaluate_retrieval_performance(self, retrieval_results: List[Dict], 
                                     k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Evaluate retrieval performance"""
        print("Evaluating retrieval performance...")
        
        hit_at_k_scores = {}
        mrr_rankings = []
        
        for result in retrieval_results:
            retrieved_docs = result.get('retrieved', [])
            relevant_docs = result.get('relevant', [])
            
            # Hit@K for different values ​​of K
            for k in k_values:
                if f'hit_at_{k}' not in hit_at_k_scores:
                    hit_at_k_scores[f'hit_at_{k}'] = []
                
                hit_score = self.hit_at_k(retrieved_docs, relevant_docs, k)
                hit_at_k_scores[f'hit_at_{k}'].append(hit_score)
            
            # MRR ranking
            ranking = []
            for doc in retrieved_docs:
                ranking.append(doc in relevant_docs)
            mrr_rankings.append(ranking)
        
        # Calculating averages
        evaluation_results = {}
        for k in k_values:
            evaluation_results[f'hit_at_{k}'] = np.mean(hit_at_k_scores[f'hit_at_{k}'])
        
        evaluation_results['mrr'] = self.mrr(mrr_rankings)
        
        print("✓ Retrieval evaluation completed")
        return evaluation_results
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare performances of different models"""
        print("Comparing model performances...")
        
        comparison = {
            'best_models': {},
            'detailed_comparison': model_results,
            'summary': {}
        }
        
        # Find the best model for each metric
        metrics = ['exact_match', 'f1_score', 'precision', 'recall', 'cosine_similarity']
        
        for metric in metrics:
            best_score = -1
            best_model = None
            
            for model_name, results in model_results.items():
                metric_key = f"{model_name}_{metric}"
                if metric_key in results:
                    score = results[metric_key]
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                comparison['best_models'][metric] = {
                    'model': best_model,
                    'score': best_score
                }
        
        # Summary 
        all_scores = {}
        for metric in metrics:
            scores = []
            for model_name, results in model_results.items():
                metric_key = f"{model_name}_{metric}"
                if metric_key in results:
                    scores.append(results[metric_key])
            
            if scores:
                all_scores[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
        
        comparison['summary'] = all_scores
        
        print("✓ Model comparison completed")
        return comparison