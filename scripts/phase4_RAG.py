#!/usr/bin/env python3
"""
Phase 4: RAG Evaluation with Multiple Embedding Models
"""

import sys
import os
import warnings
import gc
import time
from typing import List, Dict, Any

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import RAGEvaluator
from src.retrieval import RetrievalSystem
from src.llama_client import LlamaClient
from src.utils import load_config, save_results, ensure_directories
import pandas as pd

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
    
    models_dir = "models"
    trained_models = []
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path) and "finetuned" in item:
                trained_models.append(model_path)
    
    if not trained_models:
        print("⚠ No fine-tuned models found, will use base model")
        return True
    
    print(f"✓ Found {len(trained_models)} fine-tuned models")
    
    faiss_dir = "results/faiss"
    if not os.path.exists(faiss_dir):
        print("✗ FAISS indexes not found. Please run Phase 3!")
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

def find_models_and_indices():
    """Find models and indices"""
    models_dir = "models"
    faiss_dir = "results/faiss"
    
    model_paths = []
    faiss_indices = {}
    
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path) and "finetuned" in item:
                model_paths.append(model_path)
                
                model_name = os.path.basename(model_path)
                for chunk_type in ["word", "sentence"]:
                    index_file = f"{faiss_dir}/drugs_{chunk_type}_chunks.index"
                    if os.path.exists(index_file):
                        faiss_indices[model_name] = index_file
                        break
    
    if not model_paths:
        print("No fine-tuned models found, adding base models...")
        base_models = [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/distiluse-base-multilingual-cased-v2",
            "intfloat/multilingual-e5-base"
        ]
        
        for base_model in base_models:
            model_name = base_model.split('/')[-1]
            for chunk_type in ["word", "sentence"]:
                index_file = f"{faiss_dir}/drugs_{chunk_type}_chunks.index"
                if os.path.exists(index_file):
                    model_paths.append(base_model)
                    faiss_indices[model_name] = index_file
                    break
    
    return model_paths, faiss_indices

def test_llama_connection(llama_url: str = "http://127.0.0.1:8080") -> bool:
    """تست اتصال به LLaMA server"""
    print(f"Testing connection to LLaMA server at {llama_url}...")
    
    try:
        client = LlamaClient(llama_url)
        
        test_response = client.generate("سلام", max_tokens=10)
        
        if test_response:
            print("✓ LLaMA server is responding correctly")
            print(f"  Test response: {test_response[:50]}...")
            return True
        else:
            print("⚠ LLaMA server is running but not responding properly")
            return False
            
    except Exception as e:
        print(f"✗ Could not connect to LLaMA server: {e}")
        print("  Please make sure the server is running on 127.0.0.1:8080")
        return False

def run_single_model_evaluation(model_path: str, chunk_file: str, faiss_index: str,
                               test_data: List[Dict], evaluator: RAGEvaluator) -> Dict[str, Any]:
    """Single embedding evaluation in RAG"""
    model_name = os.path.basename(model_path)
    print(f"\n=== Evaluating {model_name} ===")
    
    try:
        retriever = RetrievalSystem(
            method="bm25",
            model_path=model_path,
            device="cuda" if os.path.exists("/usr/local/cuda") else "cpu"
        )
        
        if not retriever.load_chunks_and_index(chunk_file, faiss_index):
            print(f"✗ Failed to load data for {model_name}")
            return {}
        
        results = evaluator.evaluate_single_rag(
            retriever=retriever,
            test_data=test_data,
            model_name=model_name,
            sample_size=None 
        )
        
        retriever.cleanup()
        del retriever
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"✗ Error evaluating {model_name}: {e}")
        return {}

def main():
    print("=== PHASE 4: RAG Evaluation with Multiple Models ===\n")
    
    if not check_prerequisites():
        return
    
    ensure_directories()
    config = load_config()
    
    llama_url = "http://127.0.0.1:8080"
    if not test_llama_connection(llama_url):
        print("\nCannot proceed without LLaMA server. Please:")
        print("1. Make sure LLaMA server is running")
        print("2. Check that it's accessible at 127.0.0.1:8080")
        print("3. Test with a simple request first")
        return

    sample_size = config.get('evaluation', {}).get('test_size', 100)
    if isinstance(sample_size, float):
        sample_size = int(sample_size * 1000)  
    
    sample_size = min(sample_size, 200)  # max 200 question for test
    
    test_data = load_test_data(sample_size)
    if not test_data:
        print("Could not load test data!")
        return
    
    model_paths, faiss_indices = find_models_and_indices()
    
    if not model_paths:
        print("No models found for evaluation!")
        return
    
    print(f"\nFound {len(model_paths)} models to evaluate:")
    for i, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path)
        index_file = faiss_indices.get(model_name, "Not found")
        print(f"  {i+1}. {model_name}")
        print(f"     Index: {index_file}")
    
    chunk_types = []
    if os.path.exists("data/processed/drugs_word_chunks.csv"):
        chunk_types.append(("word", "data/processed/drugs_word_chunks.csv"))
    if os.path.exists("data/processed/drugs_sentence_chunks.csv"):
        chunk_types.append(("sentence", "data/processed/drugs_sentence_chunks.csv"))
    
    if not chunk_types:
        print("No chunk files found!")
        return
    
    evaluator = RAGEvaluator(llama_url)
    
    all_results = {}
    
    for chunk_type, chunk_file in chunk_types:
        print(f"\n{'='*60}")
        print(f"EVALUATING WITH {chunk_type.upper()} CHUNKS")
        print('='*60)
        
        chunk_results = {}
        
        for i, model_path in enumerate(model_paths):
            model_name = os.path.basename(model_path)
            
            print(f"\nModel {i+1}/{len(model_paths)}: {model_name}")
            print("-" * 50)
            
            index_file = None
            for chunk_t in ["word", "sentence"]:
                potential_index = f"results/faiss/drugs_{chunk_t}_chunks.index"
                if os.path.exists(potential_index):
                    index_file = potential_index
                    if chunk_t == chunk_type:  
                        break
            
            if not index_file:
                print(f"✗ No FAISS index found for {model_name}")
                continue
            
            model_results = run_single_model_evaluation(
                model_path, chunk_file, index_file, test_data, evaluator
            )
            
            if model_results:
                chunk_results.update(model_results)
                
                print(f"\n📊 {model_name} Results Summary:")
                key_metrics = ['f1_score', 'bleu_score', 'exact_match', 'success_rate', 'total_time']
                for metric in key_metrics:
                    key = f"{model_name}_{metric}"
                    if key in model_results:
                        print(f"  {metric}: {model_results[key]:.4f}")
            
            time.sleep(2)
        
        all_results[f"{chunk_type}_chunks"] = chunk_results
        
        if len(chunk_results) > 0:
            model_performances = {}
            for model_path in model_paths:
                model_name = os.path.basename(model_path)
                model_data = {k: v for k, v in chunk_results.items() if k.startswith(model_name)}
                if model_data:
                    model_performances[model_name] = model_data
            
            if model_performances:
                comparison = evaluator._analyze_model_comparison(model_performances)
                all_results[f"{chunk_type}_chunks_comparison"] = comparison
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print('='*60)
    
    for chunk_type in ["word", "sentence"]:
        comparison_key = f"{chunk_type}_chunks_comparison"
        if comparison_key in all_results:
            comparison = all_results[comparison_key]
            
            print(f"\n🏆 Best Models for {chunk_type} chunks:")
            if 'best_models' in comparison:
                for metric in ['f1_score', 'bleu_score', 'success_rate']:
                    if metric in comparison['best_models']:
                        info = comparison['best_models'][metric]
                        print(f"  {metric}: {info['model']} ({info['score']:.4f})")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"phase4_rag_evaluation_{timestamp}.json"
    
    all_results['evaluation_metadata'] = {
        'timestamp': timestamp,
        'llama_url': llama_url,
        'num_test_questions': len(test_data),
        'models_evaluated': [os.path.basename(path) for path in model_paths],
        'chunk_types': [ct[0] for ct in chunk_types],
        'sample_size': sample_size
    }
    
    evaluator.save_evaluation_results(all_results, results_file)
    
    report = evaluator.create_evaluation_report(all_results)
    report_file = f"results/phase4_rag_report_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Phase 4 completed successfully!")
    print(f"📁 Results saved to:")
    print(f"  - {results_file}")
    print(f"  - {report_file}")
    
    print(f"\n📈 Performance Summary:")
    print(f"  Total models evaluated: {len(model_paths)}")
    print(f"  Total questions processed: {len(test_data)}")
    print(f"  Chunk types tested: {len(chunk_types)}")
    
    if chunk_types:
        for chunk_type, _ in chunk_types:
            comparison_key = f"{chunk_type}_chunks_comparison"
            if comparison_key in all_results and 'best_models' in all_results[comparison_key]:
                best_f1_info = all_results[comparison_key]['best_models'].get('f1_score')
                if best_f1_info:
                    print(f"  Best {chunk_type} F1: {best_f1_info['model']} ({best_f1_info['score']:.4f})")
    
    return all_results

if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 RAG evaluation completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()