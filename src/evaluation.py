# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import List, Dict, Any, Tuple
# import torch

# class Evaluator:
#     def __init__(self):
#         pass
    
#     def exact_match(self, pred: str, gold: str) -> float:
#         """Exact Match"""
#         pred_clean = pred.strip().lower()
#         gold_clean = gold.strip().lower()
#         return float(pred_clean == gold_clean)
    
#     def f1_score(self, pred: str, gold: str) -> float:
#         """F1 Score"""
#         pred_tokens = set(pred.strip().split())
#         gold_tokens = set(gold.strip().split())
        
#         if len(pred_tokens) == 0 and len(gold_tokens) == 0:
#             return 1.0
#         if len(pred_tokens) == 0 or len(gold_tokens) == 0:
#             return 0.0
            
#         common = pred_tokens & gold_tokens
#         precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
#         recall = len(common) / len(gold_tokens) if gold_tokens else 0.0
        
#         if precision + recall == 0:
#             return 0.0
#         return 2 * precision * recall / (precision + recall)
    
#     def precision(self, pred: str, gold: str) -> float:
#         """Precision"""
#         pred_tokens = set(pred.strip().split())
#         gold_tokens = set(gold.strip().split())
        
#         if len(pred_tokens) == 0:
#             return 0.0
        
#         common = pred_tokens & gold_tokens
#         return len(common) / len(pred_tokens)
    
#     def recall(self, pred: str, gold: str) -> float:
#         """Recall"""
#         pred_tokens = set(pred.strip().split())
#         gold_tokens = set(gold.strip().split())
        
#         if len(gold_tokens) == 0:
#             return 0.0
        
#         common = pred_tokens & gold_tokens
#         return len(common) / len(gold_tokens)
    
#     def hit_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
#         """Hit@K"""
#         if not retrieved or not relevant:
#             return 0.0
        
#         top_k = retrieved[:k]
#         return float(any(item in relevant for item in top_k))
    
#     def mrr(self, rankings: List[List[bool]]) -> float:
#         """Mean Reciprocal Rank"""
#         if not rankings:
#             return 0.0
            
#         scores = []
#         for rank_list in rankings:
#             score = 0.0
#             for i, is_relevant in enumerate(rank_list):
#                 if is_relevant:
#                     score = 1.0 / (i + 1)
#                     break
#             scores.append(score)
#         return np.mean(scores)
    
#     def cosine_sim(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
#         """Cosine Similarity"""
#         if emb1.ndim == 1:
#             emb1 = emb1.reshape(1, -1)
#         if emb2.ndim == 1:
#             emb2 = emb2.reshape(1, -1)
            
#         return cosine_similarity(emb1, emb2)[0][0]
    
#     def evaluate_model_performance(self, model, test_data: List[Dict], 
#                                  model_name: str = "model") -> Dict[str, float]:
#         """Evaluate model performance on test data"""
#         print(f"Evaluating {model_name}...")
        
#         exact_matches = []
#         f1_scores = []
#         precisions = []
#         recalls = []
#         cosine_similarities = []
        
#         for i, item in enumerate(test_data):
#             if i % 100 == 0:
#                 print(f"  Processing {i}/{len(test_data)} samples...")
            
#             question = item['question']
#             gold_answer = item['answer']
            
#             question_emb = model.encode([question])
            
#             pred_answer = gold_answer  #  موقتاً برای تست
            
#             em = self.exact_match(pred_answer, gold_answer)
#             f1 = self.f1_score(pred_answer, gold_answer)
#             prec = self.precision(pred_answer, gold_answer)
#             rec = self.recall(pred_answer, gold_answer)
            
#             gold_emb = model.encode([gold_answer])
#             cos_sim = self.cosine_sim(question_emb[0], gold_emb[0])
            
#             exact_matches.append(em)
#             f1_scores.append(f1)
#             precisions.append(prec)
#             recalls.append(rec)
#             cosine_similarities.append(cos_sim)
        
#         results = {
#             f"{model_name}_exact_match": np.mean(exact_matches),
#             f"{model_name}_f1_score": np.mean(f1_scores),
#             f"{model_name}_precision": np.mean(precisions),
#             f"{model_name}_recall": np.mean(recalls),
#             f"{model_name}_cosine_similarity": np.mean(cosine_similarities),
#             f"{model_name}_num_samples": len(test_data)
#         }
        
#         print(f"✓ {model_name} evaluation completed")
#         return results
    
#     def evaluate_retrieval_performance(self, retrieval_results: List[Dict], 
#                                      k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
#         """Evaluate retrieval performance"""
#         print("Evaluating retrieval performance...")
        
#         hit_at_k_scores = {}
#         mrr_rankings = []
        
#         for result in retrieval_results:
#             retrieved_docs = result.get('retrieved', [])
#             relevant_docs = result.get('relevant', [])
            
#             # Hit@K for different values ​​of K
#             for k in k_values:
#                 if f'hit_at_{k}' not in hit_at_k_scores:
#                     hit_at_k_scores[f'hit_at_{k}'] = []
                
#                 hit_score = self.hit_at_k(retrieved_docs, relevant_docs, k)
#                 hit_at_k_scores[f'hit_at_{k}'].append(hit_score)
            
#             # MRR ranking
#             ranking = []
#             for doc in retrieved_docs:
#                 ranking.append(doc in relevant_docs)
#             mrr_rankings.append(ranking)
        
#         # Calculating averages
#         evaluation_results = {}
#         for k in k_values:
#             evaluation_results[f'hit_at_{k}'] = np.mean(hit_at_k_scores[f'hit_at_{k}'])
        
#         evaluation_results['mrr'] = self.mrr(mrr_rankings)
        
#         print("✓ Retrieval evaluation completed")
#         return evaluation_results
    
#     def compare_models(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
#         """Compare performances of different models"""
#         print("Comparing model performances...")
        
#         comparison = {
#             'best_models': {},
#             'detailed_comparison': model_results,
#             'summary': {}
#         }
        
#         # Find the best model for each metric
#         metrics = ['exact_match', 'f1_score', 'precision', 'recall', 'cosine_similarity']
        
#         for metric in metrics:
#             best_score = -1
#             best_model = None
            
#             for model_name, results in model_results.items():
#                 metric_key = f"{model_name}_{metric}"
#                 if metric_key in results:
#                     score = results[metric_key]
#                     if score > best_score:
#                         best_score = score
#                         best_model = model_name
            
#             if best_model:
#                 comparison['best_models'][metric] = {
#                     'model': best_model,
#                     'score': best_score
#                 }
        
#         # Summary 
#         all_scores = {}
#         for metric in metrics:
#             scores = []
#             for model_name, results in model_results.items():
#                 metric_key = f"{model_name}_{metric}"
#                 if metric_key in results:
#                     scores.append(results[metric_key])
            
#             if scores:
#                 all_scores[metric] = {
#                     'mean': np.mean(scores),
#                     'std': np.std(scores),
#                     'min': np.min(scores),
#                     'max': np.max(scores)
#                 }
        
#         comparison['summary'] = all_scores
        
#         print("✓ Model comparison completed")
#         return comparison

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional
import torch
import re
import os
import pandas as pd
from collections import defaultdict
import gc
import time
from sentence_transformers import SentenceTransformer

from .retrieval import RetrievalSystem, MultiModelRetrieval
from .llama_client import LlamaClient

class RAGEvaluator:
    def __init__(self, llama_url: str = "http://127.0.0.1:8080"):
        self.llama_client = LlamaClient(llama_url)
        
    def exact_match(self, pred: str, gold: str) -> float:
        """Exact Match"""
        pred_clean = self._clean_text(pred)
        gold_clean = self._clean_text(gold)
        return float(pred_clean == gold_clean)
    
    def f1_score(self, pred: str, gold: str) -> float:
        """F1 Score"""
        pred_tokens = set(self._tokenize(pred))
        gold_tokens = set(self._tokenize(gold))
        
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
        pred_tokens = set(self._tokenize(pred))
        gold_tokens = set(self._tokenize(gold))
        
        if len(pred_tokens) == 0:
            return 0.0
        
        common = pred_tokens & gold_tokens
        return len(common) / len(pred_tokens)
    
    def recall(self, pred: str, gold: str) -> float:
        """Recall"""
        pred_tokens = set(self._tokenize(pred))
        gold_tokens = set(self._tokenize(gold))
        
        if len(gold_tokens) == 0:
            return 0.0
        
        common = pred_tokens & gold_tokens
        return len(common) / len(gold_tokens)
    
    def bleu_score(self, pred: str, gold: str, n: int = 4) -> float:
        """Claculate simple BLEU score"""
        pred_tokens = self._tokenize(pred)
        gold_tokens = self._tokenize(gold)
        
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0
        
        # claculate n-gram precision
        scores = []
        for i in range(1, min(n + 1, len(pred_tokens) + 1)):
            pred_ngrams = self._get_ngrams(pred_tokens, i)
            gold_ngrams = self._get_ngrams(gold_tokens, i)
            
            if len(pred_ngrams) == 0:
                scores.append(0.0)
                continue
                
            matches = sum(min(pred_ngrams[ng], gold_ngrams[ng]) 
                         for ng in pred_ngrams if ng in gold_ngrams)
            precision = matches / len(pred_ngrams)
            scores.append(precision)
        
        if not scores or all(s == 0 for s in scores):
            return 0.0
        
        # BLEU score (geometric mean)
        bleu = np.exp(np.mean([np.log(s) if s > 0 else -float('inf') for s in scores]))
        
        # Brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(gold_tokens):
            bp = np.exp(1 - len(gold_tokens) / len(pred_tokens))
        
        return bleu * bp
    
    def rouge_l(self, pred: str, gold: str) -> float:
        """Calculate ROUGE-L score"""
        pred_tokens = self._tokenize(pred)
        gold_tokens = self._tokenize(gold)
        
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0
        
        # claculate LCS (Longest Common Subsequence)
        lcs_length = self._lcs_length(pred_tokens, gold_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def semantic_similarity(self, pred: str, gold: str, model: SentenceTransformer) -> float:
        """calculate similarity semantic"""
        try:
            embeddings = model.encode([pred, gold])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return max(0.0, similarity) 
        except:
            return 0.0
    
    def context_precision(self, retrieved_contexts: List[str], 
                         relevant_contexts: List[str]) -> float:
        """Accuracy of retrieved contexts"""
        if not retrieved_contexts:
            return 0.0
        
        relevant_count = 0
        for context in retrieved_contexts:
            if any(self._is_similar_context(context, rel_ctx) for rel_ctx in relevant_contexts):
                relevant_count += 1
        
        return relevant_count / len(retrieved_contexts)
    
    def context_recall(self, retrieved_contexts: List[str],
                      relevant_contexts: List[str]) -> float:
        """بازیابی contexts مرتبط"""
        if not relevant_contexts:
            return 1.0 
        
        retrieved_count = 0
        for rel_ctx in relevant_contexts:
            if any(self._is_similar_context(ret_ctx, rel_ctx) for ret_ctx in retrieved_contexts):
                retrieved_count += 1
        
        return retrieved_count / len(relevant_contexts)
    
    def answer_relevancy(self, answer: str, question: str, model: SentenceTransformer) -> float:
        """Answer_relevancy"""
        return self.semantic_similarity(answer, question, model)
    
    def _clean_text(self, text: str) -> str:
        """cleaning"""
        if not text:
            return ""
        
        text = text.strip().lower()
        # Remove punctuations
        text = re.sub(r'[^\w\s]', '', text)
        # Remove spaces
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenizing"""
        clean_text = self._clean_text(text)
        return clean_text.split() if clean_text else []
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """Extract n-grams"""
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Longest Common Subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _is_similar_context(self, ctx1: str, ctx2: str, threshold: float = 0.7) -> bool:
        """Checking the similarity of two contexts"""
        tokens1 = set(self._tokenize(ctx1))
        tokens2 = set(self._tokenize(ctx2))
        
        if not tokens1 or not tokens2:
            return False
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        jaccard = len(intersection) / len(union) if union else 0
        return jaccard >= threshold
    
    def evaluate_single_rag(self, retriever: RetrievalSystem, test_data: List[Dict],
                           model_name: str = "model", sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate single RAG model"""
        print(f"\n=== Evaluating RAG with {model_name} ===")
        
        if sample_size and len(test_data) > sample_size:
            test_data = test_data[:sample_size]
            print(f"Using sample of {sample_size} questions")
        
        # Metrics
        exact_matches = []
        f1_scores = []
        precisions = []
        recalls = []
        bleu_scores = []
        rouge_scores = []
        semantic_similarities = []
        context_precisions = []
        context_recalls = []
        answer_relevancies = []
        retrieval_times = []
        generation_times = []
        
        try:
            if hasattr(retriever, 'embedding_model') and retriever.embedding_model:
                eval_model = retriever.embedding_model
            else:
                eval_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        except:
            eval_model = None
            print("Warning: Could not load embedding model for semantic evaluation")
        
        failed_retrievals = 0
        failed_generations = 0
        
        for i, item in enumerate(test_data):
            if i % 50 == 0:
                print(f"  Processing {i+1}/{len(test_data)} questions...")
            
            question = item['question']
            gold_answer = item['answer']
            
            try:
                # Step 1: retrieve
                start_time = time.time()
                contexts, metadata = retriever.get_contexts_for_rag(question, top_k=5)
                retrieval_time = time.time() - start_time
                retrieval_times.append(retrieval_time)
                
                if not contexts:
                    failed_retrievals += 1
                    
                    exact_matches.append(0.0)
                    f1_scores.append(0.0)
                    precisions.append(0.0)
                    recalls.append(0.0)
                    bleu_scores.append(0.0)
                    rouge_scores.append(0.0)
                    if eval_model:
                        semantic_similarities.append(0.0)
                        answer_relevancies.append(0.0)
                    context_precisions.append(0.0)
                    context_recalls.append(0.0)
                    generation_times.append(0.0)
                    continue
                
                # Step 2: Generate response
                start_time = time.time()
                pred_answer = self.llama_client.answer_question(question, contexts)
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                if not pred_answer:
                    failed_generations += 1
                    pred_answer = ""
                
                # generation metrics calculation
                em = self.exact_match(pred_answer, gold_answer)
                f1 = self.f1_score(pred_answer, gold_answer)
                prec = self.precision(pred_answer, gold_answer)
                rec = self.recall(pred_answer, gold_answer)
                bleu = self.bleu_score(pred_answer, gold_answer)
                rouge = self.rouge_l(pred_answer, gold_answer)
                
                exact_matches.append(em)
                f1_scores.append(f1)
                precisions.append(prec)
                recalls.append(rec)
                bleu_scores.append(bleu)
                rouge_scores.append(rouge)
                
                # semantic
                if eval_model:
                    sem_sim = self.semantic_similarity(pred_answer, gold_answer, eval_model)
                    ans_rel = self.answer_relevancy(pred_answer, question, eval_model)
                    semantic_similarities.append(sem_sim)
                    answer_relevancies.append(ans_rel)
                
                # context
                ctx_precision = 1.0  # placeholder
                ctx_recall = 1.0     # placeholder
                context_precisions.append(ctx_precision)
                context_recalls.append(ctx_recall)
                
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                
                exact_matches.append(0.0)
                f1_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                bleu_scores.append(0.0)
                rouge_scores.append(0.0)
                if eval_model:
                    semantic_similarities.append(0.0)
                    answer_relevancies.append(0.0)
                context_precisions.append(0.0)
                context_recalls.append(0.0)
                retrieval_times.append(0.0)
                generation_times.append(0.0)
        
        results = {
            f"{model_name}_exact_match": np.mean(exact_matches),
            f"{model_name}_f1_score": np.mean(f1_scores),
            f"{model_name}_precision": np.mean(precisions),
            f"{model_name}_recall": np.mean(recalls),
            f"{model_name}_bleu_score": np.mean(bleu_scores),
            f"{model_name}_rouge_l": np.mean(rouge_scores),
            f"{model_name}_context_precision": np.mean(context_precisions),
            f"{model_name}_context_recall": np.mean(context_recalls),
            f"{model_name}_avg_retrieval_time": np.mean(retrieval_times),
            f"{model_name}_avg_generation_time": np.mean(generation_times),
            f"{model_name}_total_time": np.mean(retrieval_times) + np.mean(generation_times),
            f"{model_name}_failed_retrievals": failed_retrievals,
            f"{model_name}_failed_generations": failed_generations,
            f"{model_name}_success_rate": (len(test_data) - failed_retrievals - failed_generations) / len(test_data),
            f"{model_name}_num_samples": len(test_data)
        }
        
        if eval_model:
            results[f"{model_name}_semantic_similarity"] = np.mean(semantic_similarities)
            results[f"{model_name}_answer_relevancy"] = np.mean(answer_relevancies)
        
        print(f"✓ {model_name} RAG evaluation completed")
        print(f"  Success rate: {results[f'{model_name}_success_rate']:.3f}")
        print(f"  F1 Score: {results[f'{model_name}_f1_score']:.3f}")
        print(f"  BLEU Score: {results[f'{model_name}_bleu_score']:.3f}")
        print(f"  Average total time: {results[f'{model_name}_total_time']:.3f}s")
        
        return results
    
    def compare_embedding_models_in_rag(self, model_paths: List[str], chunk_file: str,
                                      faiss_indices: Dict[str, str], test_data: List[Dict],
                                      sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Comparison of embedding models in RAG"""
        print(f"\n=== Comparing {len(model_paths)} Embedding Models in RAG ===")
        
        all_results = {}
        model_performances = {}
        
        for i, model_path in enumerate(model_paths):
            model_name = os.path.basename(model_path)
            print(f"\n--- Model {i+1}/{len(model_paths)}: {model_name} ---")
            
            try:
                retriever = RetrievalSystem(
                    method="dense",
                    model_path=model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                
                faiss_index_file = faiss_indices.get(model_name)
                if not retriever.load_chunks_and_index(chunk_file, faiss_index_file):
                    print(f"✗ Failed to load data for {model_name}")
                    continue
                
                model_results = self.evaluate_single_rag(
                    retriever, test_data, model_name, sample_size
                )
                
                all_results.update(model_results)
                model_performances[model_name] = model_results
                
                retriever.cleanup()
                del retriever
                gc.collect()
                
            except Exception as e:
                print(f"✗ Error evaluating {model_name}: {e}")
                continue
        
        comparison = self._analyze_model_comparison(model_performances)
        all_results['comparison_analysis'] = comparison
        
        return all_results
    
    def _analyze_model_comparison(self, model_performances: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze model comparison"""
        if not model_performances:
            return {}
        
        metrics = [
            'exact_match', 'f1_score', 'precision', 'recall', 
            'bleu_score', 'rouge_l', 'semantic_similarity', 
            'answer_relevancy', 'success_rate', 'total_time'
        ]
        
        comparison = {
            'best_models': {},
            'ranking': {},
            'detailed_stats': {}
        }
        
        for metric in metrics:
            scores = {}
            for model_name, results in model_performances.items():
                key = f"{model_name}_{metric}"
                if key in results:
                    scores[model_name] = results[key]
            
            if scores:
                if metric == 'total_time':
                    best_model = min(scores.keys(), key=lambda x: scores[x])
                    best_score = scores[best_model]
                else:
                    best_model = max(scores.keys(), key=lambda x: scores[x])
                    best_score = scores[best_model]
                
                comparison['best_models'][metric] = {
                    'model': best_model,
                    'score': best_score
                }
                
                if metric == 'total_time':
                    sorted_models = sorted(scores.items(), key=lambda x: x[1])
                else:
                    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                comparison['ranking'][metric] = [
                    {'model': model, 'score': score} for model, score in sorted_models
                ]
                
                values = list(scores.values())
                comparison['detailed_stats'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return comparison
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str):
        """Save evaluation results"""
        import json
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy(results)
        
        filepath = f"results/{filename}"
        os.makedirs("results", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Evaluation results saved to {filepath}")
    
    def create_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Create evaluation report"""
        report = "# RAG Evaluation Report\n\n"
        
        if 'comparison_analysis' in results:
            comparison = results['comparison_analysis']
            
            report += "## Best Models by Metric\n\n"
            if 'best_models' in comparison:
                for metric, info in comparison['best_models'].items():
                    report += f"- **{metric}**: {info['model']} (Score: {info['score']:.4f})\n"
            
            report += "\n## Model Rankings\n\n"
            if 'ranking' in comparison:
                for metric in ['f1_score', 'bleu_score', 'success_rate']:
                    if metric in comparison['ranking']:
                        report += f"### {metric.replace('_', ' ').title()}\n"
                        for i, item in enumerate(comparison['ranking'][metric]):
                            report += f"{i+1}. {item['model']}: {item['score']:.4f}\n"
                        report += "\n"
        
        return report