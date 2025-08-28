#!/usr/bin/env python3
"""
Phase 1: Data Loading and Embedding Model Fine-tuning
"""

import sys
import os
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.embedding_trainer import EmbeddingTrainer
from src.utils import load_config, ensure_directories
import time

def main():
    print("=== PHASE 1: Data Loading and Embedding Training ===\n")
    
    ensure_directories()
    
    config = load_config()
    
    print("Step 1: Loading datasets...")
    data_loader = DataLoader()
    
    pquad, persian_qa = data_loader.load_datasets()
    if not pquad:
        print("Failed to load datasets!")
        return
    
    qa_data = data_loader.prepare_qa_data_for_training(pquad, persian_qa)
    
    train_data, test_data = data_loader.create_test_split(
        qa_data, test_size=config['evaluation']['test_size']
    )
    
    data_loader.save_processed_data(train_data, "train_data.csv")
    data_loader.save_processed_data(test_data, "test_data.csv")
    
    print(f"✓ Data preparation completed: {len(train_data)} train, {len(test_data)} test\n")
    
    print("Step 2: Fine-tuning embedding models...")
    
    model_results = {}
    
    for i, model_name in enumerate(config['models']):
        print(f"\n--- Training Model {i+1}/{len(config['models'])}: {model_name} ---")
        
        start_time = time.time()
        
        try:
            trainer = EmbeddingTrainer(model_name)
            
            train_examples = trainer.prepare_training_data(train_data)
            eval_examples = trainer.prepare_evaluation_data(test_data)
            
            # Fine-tuning
            model_save_path = trainer.fine_tune(
                train_examples=train_examples,
                eval_examples=None,
                epochs=1,  # limit for testing fast
                batch_size=config['evaluation']['batch_size']
            )
            
            training_time = time.time() - start_time
            
            model_results[model_name] = {
                'status': 'success',
                'save_path': model_save_path,
                'training_time': training_time,
                'num_train_examples': len(train_examples)
            }
            
            print(f"✓ {model_name} training completed in {training_time:.1f}s")
            
        except Exception as e:
            print(f"✗ Error training {model_name}: {str(e)}")
            model_results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    print("\n=== PHASE 1 RESULTS ===")
    successful_models = []
    failed_models = []
    
    for model_name, result in model_results.items():
        if result['status'] == 'success':
            successful_models.append(model_name)
            print(f"✓ {model_name}: {result['training_time']:.1f}s")
        else:
            failed_models.append(model_name)
            print(f"✗ {model_name}: {result['error']}")
    
    print(f"\nSummary: {len(successful_models)} successful, {len(failed_models)} failed")
    
    from src.utils import save_results
    save_results(model_results, "phase1_training_results.json")
    
    if successful_models:
        print(f"\n✓ Phase 1 completed successfully!")
        print(f"Trained models available for next phase: {successful_models}")
    else:
        print(f"\n✗ Phase 1 failed - no models trained successfully")
    
    return model_results

if __name__ == "__main__":
    results = main()