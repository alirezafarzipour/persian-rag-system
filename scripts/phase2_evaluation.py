#!/usr/bin/env python3
"""
Phase 2: Evaluation of Fine-tuned Embedding Models
"""

import sys
import os
import warnings
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import Evaluator
from src.utils import load_config, save_results
import time
import json

def load_test_data():
    try:
        df = pd.read_csv("data/processed/test_data.csv")
        return df.to_dict('records')
    except FileNotFoundError:
        print("Error: test_data.csv not found! Run phase1 first.")
        return None

def find_trained_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("Error: models directory not found! Run phase1 first.")
        return []
    
    trained_models = []
    for item in os.listdir(models_dir):
        model_path = os.path.join(models_dir, item)
        if os.path.isdir(model_path) and "finetuned" in item:
            trained_models.append(model_path)
    
    return trained_models

def load_model_safely(model_path: str):
    try:
        print(f"Loading model from {model_path}...")
        model = SentenceTransformer(model_path)
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

def main():
    print("=== PHASE 2: Model Evaluation ===\n")
    
    config = load_config()
    
    print("Step 1: Loading test data...")
    test_data = load_test_data()
    if not test_data:
        return
    
    print(f"✓ Loaded {len(test_data)} test samples\n")
    
    print("Step 2: Finding trained models...")
    trained_models = find_trained_models()
    if not trained_models:
        print("No trained models found! Run phase1 first.")
        return
    
    print(f"Found {len(trained_models)} trained models:")
    for model_path in trained_models:
        print(f"  - {model_path}")
    print()
    
    evaluator = Evaluator()
    
    print("Step 3: Evaluating models...")
    all_results = {}
    
    for i, model_path in enumerate(trained_models):
        model_name = os.path.basename(model_path)
        print(f"\n--- Evaluating Model {i+1}/{len(trained_models)}: {model_name} ---")
        
        start_time = time.time()
        
        model = load_model_safely(model_path)
        if not model:
            continue
        
        eval_data = test_data[:500]  # Limit for test
        # eval_data = test_data 
        
        try:
            results = evaluator.evaluate_model_performance(
                model=model,
                test_data=eval_data,
                model_name=model_name
            )
            
            eval_time = time.time() - start_time
            results[f'{model_name}_evaluation_time'] = eval_time
            
            serializable_results = {k: (v.item() if hasattr(v, 'item') else v) for k, v in results.items()}
            all_results[model_name] = serializable_results
            
            print(f"✓ {model_name} Results:")
            print(f"  - Exact Match: {results[f'{model_name}_exact_match']:.4f}")
            print(f"  - F1 Score: {results[f'{model_name}_f1_score']:.4f}")
            print(f"  - Precision: {results[f'{model_name}_precision']:.4f}")
            print(f"  - Recall: {results[f'{model_name}_recall']:.4f}")
            print(f"  - Cosine Similarity: {results[f'{model_name}_cosine_similarity']:.4f}")
            print(f"  - Evaluation Time: {eval_time:.1f}s")
            
        except Exception as e:
            print(f"✗ Error evaluating {model_name}: {e}")
            all_results[model_name] = {'status': 'failed', 'error': str(e)}
    
    print("\n" + "="*50)
    print("Step 4: Comparing models...")
    
    successful_results = {k: v for k, v in all_results.items() 
                         if 'status' not in v or v['status'] != 'failed'}
    
    if successful_results:
        comparison = evaluator.compare_models(successful_results)
        
        print("\n🏆 BEST MODELS BY METRIC:")
        for metric, info in comparison['best_models'].items():
            print(f"  {metric}: {info['model']} ({info['score']:.4f})")
        
        save_results(all_results, "phase2_evaluation_results.json")
        save_results(comparison, "phase2_model_comparison.json")
        
        print(f"\n✓ Phase 2 completed successfully!")
        print(f"✓ Evaluated {len(successful_results)} models")
        print(f"✓ Results saved to results/")
        
        print(f"\n📊 SUMMARY:")
        for metric in ['exact_match', 'f1_score', 'precision', 'recall', 'cosine_similarity']:
            if metric in comparison['summary']:
                stats = comparison['summary'][metric]
                print(f"  {metric}: avg={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    else:
        print("✗ No models evaluated successfully!")
    
    return all_results

if __name__ == "__main__":
    results = main()