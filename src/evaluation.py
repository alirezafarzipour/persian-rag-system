import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Evaluator:
    def exact_match(self, pred, gold):
        """محاسبه Exact Match"""
        return int(pred.strip() == gold.strip())
    
    def f1_score(self, pred, gold):
        """محاسبه F1 Score"""
        pred_tokens = set(pred.split())
        gold_tokens = set(gold.split())
        
        if len(pred_tokens) == 0 and len(gold_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0
            
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def hit_at_k(self, retrieved, relevant, k):
        """محاسبه Hit@K"""
        return int(any(item in relevant for item in retrieved[:k]))
    
    def mrr(self, rankings):
        """Mean Reciprocal Rank"""
        scores = []
        for rank_list in rankings:
            score = 0
            for i, item in enumerate(rank_list):
                if item:  # اگر آیتم مرتبط باشد
                    score = 1.0 / (i + 1)
                    break
            scores.append(score)
        return np.mean(scores)
    
    def cosine_sim(self, emb1, emb2):
        """محاسبه Cosine Similarity"""
        return cosine_similarity([emb1], [emb2])[0][0]