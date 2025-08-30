#!/usr/bin/env python3
"""
Script to create individual FAISS indices for each embedding model
This ensures each model uses its own embeddings in Phase 4 evaluation
"""

import os
import sys
import gc
import time
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_available_models():
    """Get list of available embedding models"""
    models = []
    
    # Check fine-tuned models
    models_dir = "models"
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path) and "finetuned" in item:
                models.append(model_path)
    
    # Add base models
    base_models = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "intfloat/multilingual-e5-base",
        # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ]
    
    models.extend(base_models)
    return models

def check_existing_indices(model_name: str) -> Dict[str, bool]:
    """Check which indices already exist for a model"""
    faiss_dir = "results/faiss"
    existing = {}
    
    for chunk_type in ["word", "sentence"]:
        index_file = f"{faiss_dir}/{model_name}_drugs_{chunk_type}_chunks.index"
        existing[chunk_type] = os.path.exists(index_file)
    
    return existing

def create_model_embeddings(model_path: str, chunk_file: str, chunk_type: str) -> bool:
    """Create embeddings for a specific model and chunk type"""
    model_name = os.path.basename(model_path)
    print(f"\n--- Creating {chunk_type} embeddings for {model_name} ---")
    
    # Check if already exists
    faiss_dir = "results/faiss"
    os.makedirs(faiss_dir, exist_ok=True)
    index_file = f"{faiss_dir}/{model_name}_drugs_{chunk_type}_chunks.index"
    
    if os.path.exists(index_file):
        print(f"✓ Index already exists: {index_file}")
        return True
    
    if not os.path.exists(chunk_file):
        print(f"✗ Chunk file not found: {chunk_file}")
        return False
    
    try:
        # Load embedding model
        print(f"Loading embedding model: {model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        embedding_model = SentenceTransformer(model_path, device=device)
        
        # Load chunks
        print(f"Loading chunks from {chunk_file}")
        df = pd.read_csv(chunk_file, encoding='utf-8')
        texts = df['text'].tolist()
        
        print(f"Processing {len(texts)} chunks...")
        
        # Create embeddings in batches to manage memory
        batch_size = 32
        embeddings = []
        
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Encode batch
            batch_embeddings = embedding_model.encode(
                batch_texts, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            embeddings.extend(batch_embeddings)
            
            # Progress reporting
            if i % (batch_size * 10) == 0:
                elapsed = time.time() - start_time
                processed = min(i + batch_size, len(texts))
                progress = processed / len(texts)
                eta = (elapsed / progress - elapsed) if progress > 0 else 0
                
                print(f"  Progress: {processed}/{len(texts)} ({progress:.1%}) - "
                      f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")
            
            # Memory management
            if i % (batch_size * 20) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Convert to numpy array
        embeddings = np.array(embeddings).astype('float32')
        print(f"Created embeddings shape: {embeddings.shape}")
        
        # Create FAISS index
        print("Creating FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use IndexFlatL2 for exact search
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Save index
        faiss.write_index(index, index_file)
        
        elapsed_total = time.time() - start_time
        print(f"✓ Successfully created and saved FAISS index: {index_file}")
        print(f"  Total time: {elapsed_total:.2f}s")
        print(f"  Index size: {index.ntotal} vectors, {dimension} dimensions")
        
        # Clean up
        del embedding_model, embeddings, index
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating embeddings for {model_name} ({chunk_type}): {e}")
        return False

def main():
    print("=== Creating Individual Model Embeddings ===\n")
    
    # Check prerequisites
    chunk_files = {
        "word": "data/processed/drugs_word_chunks.csv",
        "sentence": "data/processed/drugs_sentence_chunks.csv"
    }
    
    missing_files = [f for f in chunk_files.values() if not os.path.exists(f)]
    if missing_files:
        print("✗ Missing chunk files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Please run Phase 1 and 2 first!")
        return
    
    # Get available models
    models = get_available_models()
    if not models:
        print("✗ No embedding models found!")
        return
    
    print(f"Found {len(models)} embedding models:")
    for i, model in enumerate(models):
        model_name = os.path.basename(model)
        existing = check_existing_indices(model_name)
        status = []
        for chunk_type, exists in existing.items():
            status.append(f"{chunk_type}: {'✓' if exists else '✗'}")
        
        print(f"  {i+1}. {model_name}")
        print(f"     Existing indices: {', '.join(status)}")
    
    print(f"\nProcessing models...")
    
    total_models = len(models)
    total_tasks = total_models * 2  # word + sentence for each model
    completed_tasks = 0
    failed_tasks = 0
    
    overall_start = time.time()
    
    for i, model_path in enumerate(models):
        model_name = os.path.basename(model_path)
        
        print(f"\n{'='*60}")
        print(f"Processing Model {i+1}/{total_models}: {model_name}")
        print('='*60)
        
        # Check what needs to be created
        existing = check_existing_indices(model_name)
        
        for chunk_type, chunk_file in chunk_files.items():
            if existing[chunk_type]:
                print(f"⏭ Skipping {chunk_type} chunks (already exists)")
                completed_tasks += 1
                continue
            
            success = create_model_embeddings(model_path, chunk_file, chunk_type)
            
            if success:
                completed_tasks += 1
                print(f"✅ Completed {chunk_type} chunks for {model_name}")
            else:
                failed_tasks += 1
                print(f"❌ Failed {chunk_type} chunks for {model_name}")
            
            # Brief pause between chunks
            time.sleep(1)
        
        # Progress summary
        progress = (completed_tasks + failed_tasks) / total_tasks
        print(f"\nOverall Progress: {completed_tasks + failed_tasks}/{total_tasks} ({progress:.1%})")
        print(f"Completed: {completed_tasks}, Failed: {failed_tasks}")
        
        # Longer pause between models
        if i < len(models) - 1:
            print("Pausing before next model...")
            time.sleep(3)
    
    # Final summary
    overall_elapsed = time.time() - overall_start
    
    print(f"\n{'='*60}")
    print("EMBEDDING CREATION SUMMARY")
    print('='*60)
    
    print(f"Total time: {overall_elapsed:.2f}s ({overall_elapsed/60:.1f} minutes)")
    print(f"Models processed: {total_models}")
    print(f"Total tasks: {total_tasks}")
    print(f"Completed: {completed_tasks}")
    print(f"Failed: {failed_tasks}")
    print(f"Success rate: {completed_tasks/total_tasks:.1%}")
    
    if failed_tasks == 0:
        print("\n🎉 All embeddings created successfully!")
        print("You can now run Phase 4 evaluation with model-specific indices.")
    else:
        print(f"\n⚠ {failed_tasks} tasks failed. Check the logs above for details.")
    
    # List all created indices
    faiss_dir = "results/faiss"
    if os.path.exists(faiss_dir):
        indices = [f for f in os.listdir(faiss_dir) if f.endswith('.index')]
        model_specific = [f for f in indices if '_drugs_' in f and not f.startswith('drugs_')]
        
        print(f"\n📁 Created model-specific indices: {len(model_specific)}")
        for idx_file in sorted(model_specific):
            size_mb = os.path.getsize(os.path.join(faiss_dir, idx_file)) / 1024 / 1024
            print(f"  - {idx_file} ({size_mb:.1f} MB)")

def verify_indices():
    """Verify created indices are working"""
    print("\n=== Verifying Created Indices ===")
    
    faiss_dir = "results/faiss"
    if not os.path.exists(faiss_dir):
        print("✗ FAISS directory not found")
        return
    
    indices = [f for f in os.listdir(faiss_dir) if f.endswith('.index')]
    model_specific = [f for f in indices if '_drugs_' in f and not f.startswith('drugs_')]
    
    print(f"Verifying {len(model_specific)} model-specific indices...")
    
    for idx_file in model_specific:
        try:
            index_path = os.path.join(faiss_dir, idx_file)
            index = faiss.read_index(index_path)
            
            print(f"  ✓ {idx_file}: {index.ntotal} vectors, {index.d} dimensions")
            
            # Quick test search
            if index.ntotal > 0:
                test_vector = np.random.random((1, index.d)).astype('float32')
                distances, indices = index.search(test_vector, 1)
                print(f"    Test search: distance={distances[0][0]:.4f}")
            
        except Exception as e:
            print(f"  ✗ {idx_file}: Error loading - {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create individual model embeddings")
    parser.add_argument("--verify", action="store_true", help="Verify existing indices")
    parser.add_argument("--force", action="store_true", help="Recreate existing indices")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_indices()
    else:
        try:
            main()
            verify_indices()
        except KeyboardInterrupt:
            print("\n⚠ Process interrupted by user")
        except Exception as e:
            print(f"\n✗ Process failed: {e}")
            import traceback
            traceback.print_exc()