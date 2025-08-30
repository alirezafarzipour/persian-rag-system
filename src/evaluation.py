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
        """Exact Match with improved text cleaning"""
        pred_clean = self._clean_text(pred)
        gold_clean = self._clean_text(gold)
        return float(pred_clean == gold_clean)
    
    def f1_score(self, pred: str, gold: str) -> float:
        """F1 Score with better tokenization"""
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
        """Calculate enhanced BLEU score"""
        pred_tokens = self._tokenize(pred)
        gold_tokens = self._tokenize(gold)
        
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0
        
        # Calculate n-gram precision
        scores = []
        for i in range(1, min(n + 1, len(pred_tokens) + 1)):
            pred_ngrams = self._get_ngrams(pred_tokens, i)
            gold_ngrams = self._get_ngrams(gold_tokens, i)
            
            if len(pred_ngrams) == 0:
                scores.append(0.0)
                continue
                
            matches = sum(min(pred_ngrams[ng], gold_ngrams[ng]) 
                         for ng in pred_ngrams if ng in gold_ngrams)
            precision = matches / sum(pred_ngrams.values())
            scores.append(precision)
        
        if not scores or all(s == 0 for s in scores):
            return 0.0
        
        # BLEU score (geometric mean)
        bleu = np.exp(np.mean([np.log(s) if s > 0 else -float('inf') for s in scores]))
        
        # Brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(gold_tokens):
            bp = np.exp(1 - len(gold_tokens) / len(pred_tokens))
        
        return min(bleu * bp, 1.0)  # Cap at 1.0
    
    def rouge_l(self, pred: str, gold: str) -> float:
        """Calculate ROUGE-L score"""
        pred_tokens = self._tokenize(pred)
        gold_tokens = self._tokenize(gold)
        
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0
        
        # Calculate LCS (Longest Common Subsequence)
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
        """Calculate semantic similarity with error handling"""
        try:
            if not pred.strip() or not gold.strip():
                return 0.0
            
            embeddings = model.encode([pred, gold], show_progress_bar=False)
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return max(0.0, min(1.0, similarity))  # Ensure [0,1] range
        except Exception as e:
            print(f"Warning: Semantic similarity calculation failed: {e}")
            return 0.0
    
    def context_precision(self, retrieved_contexts: List[str], 
                         relevant_contexts: List[str]) -> float:
        """Precision of retrieved contexts"""
        if not retrieved_contexts:
            return 0.0
        
        relevant_count = 0
        for context in retrieved_contexts:
            if any(self._is_similar_context(context, rel_ctx) for rel_ctx in relevant_contexts):
                relevant_count += 1
        
        return relevant_count / len(retrieved_contexts)
    
    def context_recall(self, retrieved_contexts: List[str],
                      relevant_contexts: List[str]) -> float:
        """Recall of relevant contexts"""
        if not relevant_contexts:
            return 1.0 
        
        retrieved_count = 0
        for rel_ctx in relevant_contexts:
            if any(self._is_similar_context(ret_ctx, rel_ctx) for ret_ctx in retrieved_contexts):
                retrieved_count += 1
        
        return retrieved_count / len(relevant_contexts)
    
    def answer_relevancy(self, answer: str, question: str, model: SentenceTransformer) -> float:
        """Answer relevancy to question"""
        return self.semantic_similarity(answer, question, model)
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not text:
            return ""
        
        text = text.strip().lower()
        
        # Remove Persian-specific artifacts
        text = re.sub(r'[۰-۹]', lambda x: str(ord(x.group()) - ord('۰')), text)  # Persian to English digits
        
        # Remove punctuations but keep Persian characters
        text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization"""
        clean_text = self._clean_text(text)
        if not clean_text:
            return []
        
        # Simple word-based tokenization that works well for Persian
        tokens = clean_text.split()
        
        # Filter out very short tokens and common stopwords
        persian_stopwords = {'در', 'از', 'به', 'با', 'که', 'را', 'و', 'تا', 'بر', 'این', 'آن'}
        tokens = [token for token in tokens if len(token) > 1 and token not in persian_stopwords]
        
        return tokens
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """Extract n-grams"""
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return dict(ngrams)
    
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
        """Check similarity of two contexts"""
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
        """Enhanced single RAG model evaluation"""
        print(f"\n=== Evaluating RAG with {model_name} ===")
        
        if sample_size and len(test_data) > sample_size:
            test_data = test_data[:sample_size]
            print(f"Using sample of {sample_size} questions")
        
        # Initialize metrics
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
        
        # Load evaluation model for semantic metrics
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
            if i % 25 == 0:
                print(f"  Processing {i+1}/{len(test_data)} questions...")
            
            question = item['question']
            gold_answer = item['answer']
            
            try:
                # Step 1: Retrieve contexts
                start_time = time.time()
                contexts, metadata = retriever.get_contexts_for_rag(question, top_k=5)
                retrieval_time = time.time() - start_time
                retrieval_times.append(retrieval_time)
                
                if not contexts:
                    failed_retrievals += 1
                    self._add_zero_scores(exact_matches, f1_scores, precisions, recalls, 
                                        bleu_scores, rouge_scores, semantic_similarities,
                                        context_precisions, context_recalls, answer_relevancies,
                                        generation_times, eval_model)
                    continue
                
                # Step 2: Generate response with enhanced cleaning
                start_time = time.time()
                pred_answer = self.llama_client.answer_question(question, contexts)
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                if not pred_answer or not pred_answer.strip():
                    failed_generations += 1
                    pred_answer = ""
                
                # Calculate generation metrics
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
                
                # Semantic metrics
                if eval_model:
                    sem_sim = self.semantic_similarity(pred_answer, gold_answer, eval_model)
                    ans_rel = self.answer_relevancy(pred_answer, question, eval_model)
                    semantic_similarities.append(sem_sim)
                    answer_relevancies.append(ans_rel)
                
                # Context metrics (placeholder - can be enhanced with ground truth)
                ctx_precision = 1.0  # Placeholder
                ctx_recall = 1.0     # Placeholder
                context_precisions.append(ctx_precision)
                context_recalls.append(ctx_recall)
                
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                self._add_zero_scores(exact_matches, f1_scores, precisions, recalls, 
                                    bleu_scores, rouge_scores, semantic_similarities,
                                    context_precisions, context_recalls, answer_relevancies,
                                    generation_times, eval_model)
                retrieval_times.append(0.0)
        
        # Compile results
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
        if eval_model:
            print(f"  Semantic Similarity: {results[f'{model_name}_semantic_similarity']:.3f}")
        print(f"  Average total time: {results[f'{model_name}_total_time']:.3f}s")
        
        return results
    
    def _add_zero_scores(self, *score_lists, eval_model=None):
        """Helper to add zero scores for failed cases"""
        for score_list in score_lists[:-2]:  # All except generation_times and eval_model
            score_list.append(0.0)
        
        # generation_times
        score_lists[-2].append(0.0)
        
        # Handle semantic scores if eval_model exists
        if eval_model and len(score_lists) > 10:
            score_lists[6].append(0.0)  # semantic_similarities
            score_lists[9].append(0.0)  # answer_relevancies
    
    def _analyze_model_comparison(self, model_performances: Dict[str, Dict]) -> Dict[str, Any]:
        """Enhanced model comparison analysis"""
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
            'detailed_stats': {},
            'performance_summary': {}
        }
        
        for metric in metrics:
            scores = {}
            for model_name, results in model_performances.items():
                key = f"{model_name}_{metric}"
                if key in results:
                    scores[model_name] = results[key]
            
            if scores:
                # Find best model (lowest for time, highest for others)
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
                
                # Create ranking
                if metric == 'total_time':
                    sorted_models = sorted(scores.items(), key=lambda x: x[1])
                else:
                    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                comparison['ranking'][metric] = [
                    {'model': model, 'score': score} for model, score in sorted_models
                ]
                
                # Calculate statistics
                values = list(scores.values())
                comparison['detailed_stats'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values)
                }
        
        # Overall performance summary
        if model_performances:
            comparison['performance_summary'] = {
                'total_models': len(model_performances),
                'metrics_evaluated': len([m for m in metrics if m in comparison['best_models']]),
            }
        
        return comparison
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str):
        """Save evaluation results with better error handling"""
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
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, ensure_ascii=False, indent=2)
            print(f"✓ Evaluation results saved to {filepath}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def create_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Create enhanced evaluation report"""
        report = "# Enhanced RAG Evaluation Report\n\n"
        
        # Add timestamp and metadata
        if 'evaluation_metadata' in results:
            metadata = results['evaluation_metadata']
            report += "## Evaluation Metadata\n\n"
            report += f"- **Timestamp**: {metadata.get('timestamp', 'N/A')}\n"
            report += f"- **Models Evaluated**: {len(metadata.get('models_evaluated', []))}\n"
            report += f"- **Test Questions**: {metadata.get('num_test_questions', 'N/A')}\n"
            report += f"- **Chunk Types**: {', '.join(metadata.get('chunk_types', []))}\n"
            report += f"- **Enhancement**: {metadata.get('enhancement', 'N/A')}\n\n"
        
        # Best models by metric
        for chunk_type in ["word", "sentence"]:
            comparison_key = f"{chunk_type}_chunks_comparison"
            if comparison_key in results:
                comparison = results[comparison_key]
                
                report += f"## Best Models for {chunk_type.title()} Chunks\n\n"
                if 'best_models' in comparison:
                    for metric, info in comparison['best_models'].items():
                        report += f"- **{metric.replace('_', ' ').title()}**: {info['model']} (Score: {info['score']:.4f})\n"
                
                # Add detailed rankings
                report += f"\n### Detailed Rankings for {chunk_type.title()} Chunks\n\n"
                if 'ranking' in comparison:
                    for metric in ['f1_score', 'bleu_score', 'success_rate', 'total_time']:
                        if metric in comparison['ranking']:
                            report += f"#### {metric.replace('_', ' ').title()}\n"
                            for i, item in enumerate(comparison['ranking'][metric]):
                                report += f"{i+1}. {item['model']}: {item['score']:.4f}\n"
                            report += "\n"
                
                # Add performance statistics
                if 'detailed_stats' in comparison:
                    report += f"### Performance Statistics for {chunk_type.title()} Chunks\n\n"
                    report += "| Metric | Mean | Std | Min | Max | Range |\n"
                    report += "|--------|------|-----|-----|-----|-------|\n"
                    
                    for metric, stats in comparison['detailed_stats'].items():
                        if metric in ['f1_score', 'bleu_score', 'success_rate']:
                            report += f"| {metric.replace('_', ' ').title()} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['range']:.4f} |\n"
                    
                    report += "\n"
        
        return report