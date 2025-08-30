#!/usr/bin/env python3
"""
Phase 4: RAG Evaluation with BM25 and TF-IDF Methods
Updated version to evaluate BM25 and TF-IDF retrieval methods instead of dense
"""

import sys
import os
import warnings
import gc
import time
import re
from typing import List, Dict, Any

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import RAGEvaluator
from src.retrieval import RetrievalSystem
from src.llama_client import LlamaClient
from src.utils import load_config, save_results, ensure_directories
import pandas as pd

def clean_prediction(text):
    """Clean model prediction from artifacts and extra text."""
    if not text:
        return ""
    
    # Remove special tokens
    text = re.sub(r'<\|[^|]*\|>', '', text)
    text = re.sub(r'user[a-zA-Z]*', '', text)
    text = re.sub(r'assistant[a-zA-Z]*', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If prediction is too long, take first meaningful part
    sentences = text.split('.')
    if len(sentences) > 1:
        # Take first sentence if it's not empty
        first_sentence = sentences[0].strip()
        if first_sentence:
            text = first_sentence
    
    # If still too long, take first 50 characters
    if len(text) > 50:
        text = text[:50].strip()
    
    return text

def check_prerequisites():
    """Check prerequisites"""
    print("Checking prerequisites...")
    
    required_files = [
        "data/processed/test_data.csv",
        "data/processed/drugs_word_chunks.csv",
        "data/processed/drugs_sentence_chunks.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("✗ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run Phase 1, 2, and 3 first!")
        return False
    
    print("✓ All prerequisites check passed")
    return True

def load_test_data(sample_size: int = None) -> List[Dict]:
    """Load test data"""
    print("Loading test data...")
    
    try:
        df = pd.read_csv("data/processed/test_data.csv", encoding='utf-8')
        test_data = df.to_dict('records')
        
        if sample_size and len(test_data) > sample_size:
            test_data = test_data[:sample_size]
            print(f"Using sample of {sample_size} questions from {len(df)} total")
        else:
            print(f"Loaded {len(test_data)} test questions")
        
        return test_data
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

def test_llama_connection(llama_url: str = "http://127.0.0.1:8080") -> bool:
    """Test connection to LLaMA server"""
    print(f"Testing connection to LLaMA server at {llama_url}...")
    
    try:
        client = LlamaClient(llama_url)
        test_response = client.generate("سلام", max_tokens=10)
        
        if test_response:
            cleaned_response = clean_prediction(test_response)
            print("✓ LLaMA server is responding correctly")
            print(f"  Test response: {cleaned_response[:50]}...")
            return True
        else:
            print("⚠ LLaMA server is running but not responding properly")
            return False
            
    except Exception as e:
        print(f"✗ Could not connect to LLaMA server: {e}")
        print("  Please make sure the server is running on 127.0.0.1:8080")
        return False

def run_single_method_evaluation(method: str, chunk_file: str, test_data: List[Dict], 
                               evaluator: RAGEvaluator) -> Dict[str, Any]:
    """Single retrieval method evaluation in RAG"""
    print(f"\n=== Evaluating {method.upper()} Method ===")
    
    try:
        # Create retriever with specific method (no embedding model needed for BM25/TF-IDF)
        retriever = RetrievalSystem(
            method=method,
            model_path=None,  # Not needed for BM25/TF-IDF
            device="cuda" if os.path.exists("/usr/local/cuda") else "cpu"
        )
        
        # For BM25 and TF-IDF, we don't need FAISS index
        if not retriever.load_chunks_and_index(chunk_file, faiss_index_file=None):
            print(f"✗ Failed to load data for {method}")
            return {}
        
        # Evaluate the method
        results = evaluator.evaluate_single_rag(
            retriever=retriever,
            test_data=test_data,
            model_name=method,
            sample_size=200
        )
        
        # Clean up
        retriever.cleanup()
        del retriever
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"✗ Error evaluating {method}: {e}")
        return {}

def main():
    print("=== PHASE 4: RAG Evaluation with BM25 and TF-IDF Methods ===\n")
    
    if not check_prerequisites():
        return
    
    ensure_directories()
    config = load_config()
    
    # Test LLaMA connection
    llama_url = "http://127.0.0.1:8080"
    if not test_llama_connection(llama_url):
        print("\nCannot proceed without LLaMA server. Please:")
        print("1. Make sure LLaMA server is running")
        print("2. Check that it's accessible at 127.0.0.1:8080")
        print("3. Test with a simple request first")
        return

    # Load test data
    sample_size = config.get('evaluation', {}).get('test_size', 1000)
    if isinstance(sample_size, float):
        sample_size = int(sample_size * 1000)
    
    sample_size = min(sample_size, 500)  # Max 200 questions for test
    
    test_data = load_test_data(sample_size)
    if not test_data:
        print("Could not load test data!")
        return
    
    # Define retrieval methods to evaluate
    methods_to_evaluate = ["bm25", "tfidf"]
    
    print(f"\nMethods to evaluate: {', '.join([m.upper() for m in methods_to_evaluate])}")
    
    # Determine available chunk types
    chunk_types = []
    if os.path.exists("data/processed/drugs_word_chunks.csv"):
        chunk_types.append(("word", "data/processed/drugs_word_chunks.csv"))
    if os.path.exists("data/processed/drugs_sentence_chunks.csv"):
        chunk_types.append(("sentence", "data/processed/drugs_sentence_chunks.csv"))
    
    if not chunk_types:
        print("No chunk files found!")
        return
    
    # Create evaluator
    evaluator = RAGEvaluator(llama_url)
    
    all_results = {}
    
    # Evaluate each chunk type
    for chunk_type, chunk_file in chunk_types:
        print(f"\n{'='*60}")
        print(f"EVALUATING WITH {chunk_type.upper()} CHUNKS")
        print('='*60)
        
        chunk_results = {}
        
        for i, method in enumerate(methods_to_evaluate):
            print(f"\nMethod {i+1}/{len(methods_to_evaluate)}: {method.upper()}")
            print("-" * 50)
            
            # Run evaluation for this method
            method_results = run_single_method_evaluation(
                method, chunk_file, test_data, evaluator
            )
            
            if method_results:
                chunk_results.update(method_results)
                
                print(f"\n📊 {method.upper()} Results Summary:")
                key_metrics = ['f1_score', 'bleu_score', 'exact_match', 'success_rate', 'total_time']
                for metric in key_metrics:
                    key = f"{method}_{metric}"
                    if key in method_results:
                        print(f"  {metric}: {method_results[key]:.4f}")
            
            time.sleep(2)  # Brief pause between methods
        
        # Store results for this chunk type
        all_results[f"{chunk_type}_chunks"] = chunk_results
        
        # Analyze comparison for this chunk type
        if len(chunk_results) > 0:
            method_performances = {}
            for method in methods_to_evaluate:
                method_data = {k: v for k, v in chunk_results.items() if k.startswith(method)}
                if method_data:
                    method_performances[method] = method_data
            
            if method_performances:
                comparison = evaluator._analyze_model_comparison(method_performances)
                all_results[f"{chunk_type}_chunks_comparison"] = comparison
    
    # Final results summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print('='*60)
    
    for chunk_type in ["word", "sentence"]:
        comparison_key = f"{chunk_type}_chunks_comparison"
        if comparison_key in all_results:
            comparison = all_results[comparison_key]
            
            print(f"\n🏆 Best Methods for {chunk_type} chunks:")
            if 'best_models' in comparison:
                for metric in ['f1_score', 'bleu_score', 'success_rate']:
                    if metric in comparison['best_models']:
                        info = comparison['best_models'][metric]
                        print(f"  {metric}: {info['model']} ({info['score']:.4f})")
    
    # Compare BM25 vs TF-IDF across chunk types
    print(f"\n{'='*60}")
    print("BM25 vs TF-IDF COMPARISON")
    print('='*60)
    
    for method in methods_to_evaluate:
        print(f"\n📈 {method.upper()} Performance:")
        for chunk_type in ["word", "sentence"]:
            chunk_key = f"{chunk_type}_chunks"
            if chunk_key in all_results:
                chunk_results = all_results[chunk_key]
                f1_key = f"{method}_f1_score"
                bleu_key = f"{method}_bleu_score"
                success_key = f"{method}_success_rate"
                
                if f1_key in chunk_results:
                    print(f"  {chunk_type} chunks - F1: {chunk_results[f1_key]:.4f}, "
                          f"BLEU: {chunk_results[bleu_key]:.4f}, "
                          f"Success: {chunk_results[success_key]:.4f}")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"phase4_bm25_tfidf_evaluation_{timestamp}.json"
    
    all_results['evaluation_metadata'] = {
        'timestamp': timestamp,
        'llama_url': llama_url,
        'num_test_questions': len(test_data),
        'methods_evaluated': methods_to_evaluate,
        'chunk_types': [ct[0] for ct in chunk_types],
        'sample_size': sample_size,
        'evaluation_type': 'bm25_tfidf_comparison'
    }
    
    evaluator.save_evaluation_results(all_results, results_file)
    
    # Create report
    report = evaluator.create_evaluation_report(all_results)
    report_file = f"results/phase4_bm25_tfidf_report_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Phase 4 BM25/TF-IDF evaluation completed successfully!")
    print(f"📁 Results saved to:")
    print(f"  - {results_file}")
    print(f"  - {report_file}")
    
    print(f"\n📈 Performance Summary:")
    print(f"  Total methods evaluated: {len(methods_to_evaluate)}")
    print(f"  Total questions processed: {len(test_data)}")
    print(f"  Chunk types tested: {len(chunk_types)}")
    
    # Show best performing method for each chunk type
    for chunk_type in ["word", "sentence"]:
        comparison_key = f"{chunk_type}_chunks_comparison"
        if comparison_key in all_results and 'best_models' in all_results[comparison_key]:
            best_f1_info = all_results[comparison_key]['best_models'].get('f1_score')
            if best_f1_info:
                print(f"  Best {chunk_type} F1: {best_f1_info['model']} ({best_f1_info['score']:.4f})")
    
    return all_results

if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 BM25 & TF-IDF RAG evaluation completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()