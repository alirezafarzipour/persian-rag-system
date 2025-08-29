#!/usr/bin/env python3
"""
Phase 3: PDF Chunking and Vector Database Setup (Memory Optimized)
"""

import sys
import os
import warnings
import numpy as np
import faiss
import time
import gc
import torch
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.chunking import TextChunker
from src.utils import load_config, save_results, ensure_directories

def check_gpu_availability():
    """check GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        
        torch.cuda.empty_cache()
        return device
    else:
        print("⚠ GPU not available, using CPU")
        return torch.device('cpu')

def setup_faiss_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """Setup FAISS index"""
    print(f"Setting up FAISS index for {embeddings.shape[0]} embeddings...")
    
    dimension = embeddings.shape[1]
    
    if index_type == "flat" or embeddings.shape[0] < 1000:
        # Simple flat index
        index = faiss.IndexFlatL2(dimension)
    else:
        # IVF index 
        nlist = min(100, max(10, embeddings.shape[0] // 20))
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, nlist)
        
        print("  Training FAISS index...")
        training_data = embeddings[:min(10000, len(embeddings))].astype('float32')
        index.train(training_data)
        del training_data
        gc.collect()
    
    batch_size = 1000
    print("  Adding embeddings to index...")
    for i in range(0, embeddings.shape[0], batch_size):
        end_idx = min(i + batch_size, embeddings.shape[0])
        batch = embeddings[i:end_idx].astype('float32')
        index.add(batch)
        
        if i % (batch_size * 5) == 0 and i > 0:
            print(f"    Added {end_idx}/{embeddings.shape[0]} embeddings")
            gc.collect()
    
    print("✓ FAISS index created successfully")
    return index

def setup_chroma_collection(chunks: list, embeddings: np.ndarray, collection_name: str):
    """Setup Chroma collection"""
    try:
        import chromadb
        
        print(f"Setting up Chroma collection: {collection_name}")
        
        client = chromadb.PersistentClient(path="./chroma_db")
        
        try:
            client.delete_collection(collection_name)
            print("  Previous collection deleted")
        except:
            pass
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        batch_size = 500
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"  Adding {len(chunks)} documents in {total_batches} batches...")
        
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            
            batch_chunks = chunks[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            
            ids = [chunk['id'] for chunk in batch_chunks]
            texts = [chunk['text'] for chunk in batch_chunks]
            metadatas = [{'chunk_type': chunk['chunk_type'], 'num_words': chunk.get('num_words', 0)} 
                        for chunk in batch_chunks]
            
            collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            if i % (batch_size * 2) == 0 and i > 0:
                print(f"    Batch {(i//batch_size)+1}/{total_batches} completed")
            
            del batch_chunks, batch_embeddings, ids, texts, metadatas
            gc.collect()
        
        print(f"✓ Chroma collection '{collection_name}' created with {len(chunks)} items")
        return collection
        
    except ImportError:
        print("Warning: ChromaDB not installed. Skipping Chroma setup.")
        return None
    except Exception as e:
        print(f"Error setting up Chroma: {e}")
        return None

def generate_embeddings_with_batching(model, texts: list, batch_size: int = 16, device=None):
    """Generate embeddings"""
    print(f"Generating embeddings for {len(texts)} texts...")
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    if device and hasattr(model, '_modules'):
        try:
            model = model.to(device)
        except:
            print("Warning: Could not move model to GPU, using CPU")
            device = torch.device('cpu')
    
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        batch_texts = texts[i:end_idx]
        
        try:
            batch_embeddings = model.encode(
                batch_texts, 
                batch_size=batch_size,
                show_progress_bar=False,
                device=str(device) if device else None,
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)
            
            if (i // batch_size + 1) % 20 == 0:
                print(f"  Processed {i + len(batch_texts)}/{len(texts)} texts")
                
        except Exception as e:
            print(f"Warning: Error in batch {i//batch_size + 1}: {e}")
            try:
                batch_embeddings = model.encode(
                    batch_texts, 
                    batch_size=max(1, batch_size//4),
                    show_progress_bar=False,
                    device='cpu',
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
            except Exception as e2:
                print(f"Critical error in batch {i//batch_size + 1}: {e2}")
                dummy_embedding = np.zeros((len(batch_texts), 384))  # default dimension
                all_embeddings.append(dummy_embedding)
        
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
        
        if i % (batch_size * 10) == 0:
            gc.collect()
    
    try:
        final_embeddings = np.vstack(all_embeddings)
    except Exception as e:
        print(f"Error concatenating embeddings: {e}")
        total_samples = sum(emb.shape[0] for emb in all_embeddings)
        dimension = all_embeddings[0].shape[1] if all_embeddings else 384
        final_embeddings = np.zeros((total_samples, dimension))
        
        current_idx = 0
        for emb_batch in all_embeddings:
            batch_size = emb_batch.shape[0]
            final_embeddings[current_idx:current_idx + batch_size] = emb_batch
            current_idx += batch_size
    
    del all_embeddings
    gc.collect()
    
    print(f"✓ Generated embeddings shape: {final_embeddings.shape}")
    return final_embeddings

def main():
    print("=== PHASE 3: PDF Chunking and Vector DB (Memory Optimized) ===\n")
    
    device = check_gpu_availability()
    
    ensure_directories()
    config = load_config()
    
    pdf_path = "data/raw/Drugs.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
        print("Please place Drugs.pdf in data/raw/ directory")
        return
    
    print("Step 1: Extracting text from PDF...")
    data_loader = DataLoader()
    
    try:
        pdf_text = data_loader.extract_pdf(pdf_path)
        print(f"✓ Extracted {len(pdf_text)} characters from PDF")
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return
    
    if len(pdf_text) > 500000:
        print("⚠ Large PDF detected, will use smaller batch sizes")
        embedding_batch_size = 8
    else:
        embedding_batch_size = 16
    
    # Chunking
    print("\nStep 2: Chunking PDF text...")
    chunker = TextChunker(config)
    
    try:
        word_chunks, sentence_chunks = chunker.process_pdf_document(pdf_text)
    except Exception as e:
        print(f"Error during chunking: {e}")
        return
    
    del pdf_text
    gc.collect()
    
    # Results
    if not word_chunks and not sentence_chunks:
        print("Error: No chunks were created!")
        return
    
    print(f"\nChunking Results:")
    print(f"  Word chunks: {len(word_chunks)}")
    print(f"  Sentence chunks: {len(sentence_chunks)}")
    
    # store chunks
    if word_chunks:
        chunker.save_chunks(word_chunks, "drugs_word_chunks.csv")
    if sentence_chunks:
        chunker.save_chunks(sentence_chunks, "drugs_sentence_chunks.csv")
    
    # chunks statistics
    word_stats = chunker.get_chunk_statistics(word_chunks) if word_chunks else {}
    sentence_stats = chunker.get_chunk_statistics(sentence_chunks) if sentence_chunks else {}
    
    print(f"\n📊 CHUNKING STATISTICS:")
    if word_stats:
        print(f"Word-based chunks: {word_stats}")
    if sentence_stats:
        print(f"Sentence-based chunks: {sentence_stats}")
    
    print("\nStep 3: Loading best embedding model...")
    
    models_dir = "models"
    trained_models = []
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path) and "finetuned" in item:
                trained_models.append(model_path)
    
    if not trained_models:
        print("No trained models found! Using base model...")
        model_name_or_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model_name = "base_model"
    else:
        model_name_or_path = trained_models[0]
        model_name = os.path.basename(model_name_or_path)
        print(f"✓ Using trained model: {model_name}")
    
    try:
        model = SentenceTransformer(model_name_or_path, device=str(device))
        print(f"✓ Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model on GPU, using CPU: {e}")
        try:
            model = SentenceTransformer(model_name_or_path, device='cpu')
            device = torch.device('cpu')
            print(f"✓ Model loaded on CPU")
        except Exception as e2:
            print(f"Error loading model: {e2}")
            return
    
    print("\nStep 4: Generating embeddings with memory optimization...")
    
    word_embeddings = None
    sentence_embeddings = None
    
    # Word-based embeddings
    if word_chunks:
        print("  Generating word-based embeddings...")
        try:
            word_texts = [chunk['text'] for chunk in word_chunks]
            word_embeddings = generate_embeddings_with_batching(
                model, word_texts, batch_size=embedding_batch_size, device=device
            )
            
            del word_texts
            gc.collect()
            
        except Exception as e:
            print(f"Error generating word embeddings: {e}")
            word_embeddings = None
    
    # Sentence-based embeddings
    if sentence_chunks:
        print("  Generating sentence-based embeddings...")
        try:
            sentence_texts = [chunk['text'] for chunk in sentence_chunks]
            sentence_embeddings = generate_embeddings_with_batching(
                model, sentence_texts, batch_size=embedding_batch_size, device=device
            )
            
            del sentence_texts
            gc.collect()
            
        except Exception as e:
            print(f"Error generating sentence embeddings: {e}")
            sentence_embeddings = None
    
    if word_embeddings is not None:
        print(f"✓ Generated word embeddings: {word_embeddings.shape}")
    if sentence_embeddings is not None:
        print(f"✓ Generated sentence embeddings: {sentence_embeddings.shape}")
    
    # FAISS indexes
    print("\nStep 5: Setting up FAISS indexes...")
    
    start_time = time.time()
    
    word_faiss_index = None
    sentence_faiss_index = None
    
    # FAISS for word chunks
    if word_embeddings is not None and len(word_chunks) > 0:
        print("  Creating FAISS index for word-based chunks...")
        try:
            index_type = "ivf" if word_embeddings.shape[0] > 1000 else "flat"
            word_faiss_index = setup_faiss_index(word_embeddings, index_type)
        except Exception as e:
            print(f"Error creating word FAISS index: {e}")
    
    # FAISS for sentence chunks
    if sentence_embeddings is not None and len(sentence_chunks) > 0:
        print("  Creating FAISS index for sentence-based chunks...")
        try:
            index_type = "ivf" if sentence_embeddings.shape[0] > 1000 else "flat"
            sentence_faiss_index = setup_faiss_index(sentence_embeddings, index_type)
        except Exception as e:
            print(f"Error creating sentence FAISS index: {e}")
    
    faiss_time = time.time() - start_time
    
    # store FAISS indexes
    os.makedirs("results/faiss", exist_ok=True)
    if word_faiss_index is not None:
        faiss.write_index(word_faiss_index, "results/faiss/drugs_word_chunks.index")
        print("✓ Word FAISS index saved")
    if sentence_faiss_index is not None:
        faiss.write_index(sentence_faiss_index, "results/faiss/drugs_sentence_chunks.index")
        print("✓ Sentence FAISS index saved")
    
    print(f"✓ FAISS setup completed in {faiss_time:.2f}s")
    
    # Chroma collections
    print("\nStep 6: Setting up Chroma collections...")
    
    start_time = time.time()
    
    word_chroma = None
    sentence_chroma = None
    
    # Chroma for word chunks
    if word_chunks and word_embeddings is not None:
        print("  Creating Chroma collection for word-based chunks...")
        try:
            word_chroma = setup_chroma_collection(
                word_chunks, word_embeddings, "drugs_word_chunks"
            )
            gc.collect()
        except Exception as e:
            print(f"Error creating word Chroma collection: {e}")
    
    # Chroma for sentence chunks
    if sentence_chunks and sentence_embeddings is not None:
        print("  Creating Chroma collection for sentence-based chunks...")
        try:
            sentence_chroma = setup_chroma_collection(
                sentence_chunks, sentence_embeddings, "drugs_sentence_chunks"
            )
            gc.collect()
        except Exception as e:
            print(f"Error creating sentence Chroma collection: {e}")
    
    chroma_time = time.time() - start_time
    print(f"✓ Chroma setup completed in {chroma_time:.2f}s")
    
    print("\nStep 7: Testing index quality...")
    
    test_query = "دارو چیست؟"
    
    word_distances = None
    sentence_distances = None
    
    if word_faiss_index is not None:
        try:
            test_embedding = model.encode([test_query], device=str(device))
            word_distances, word_indices = word_faiss_index.search(
                test_embedding.astype('float32'), k=min(3, len(word_chunks))
            )
            print(f"✓ Word FAISS search test completed")
            print(f"  Top 3 distances: {word_distances[0][:3]}")
        except Exception as e:
            print(f"Error testing word FAISS: {e}")
    
    if sentence_faiss_index is not None:
        try:
            test_embedding = model.encode([test_query], device=str(device))
            sentence_distances, sentence_indices = sentence_faiss_index.search(
                test_embedding.astype('float32'), k=min(3, len(sentence_chunks))
            )
            print(f"✓ Sentence FAISS search test completed")
            print(f"  Top 3 distances: {sentence_distances[0][:3]}")
        except Exception as e:
            print(f"Error testing sentence FAISS: {e}")
    
    if word_chroma:
        try:
            word_chroma_results = word_chroma.query(
                query_texts=[test_query],
                n_results=min(3, len(word_chunks))
            )
            print(f"✓ Word Chroma search test completed")
        except Exception as e:
            print(f"Error testing word Chroma: {e}")
    
    if sentence_chroma:
        try:
            sentence_chroma_results = sentence_chroma.query(
                query_texts=[test_query], 
                n_results=min(3, len(sentence_chunks))
            )
            print(f"✓ Sentence Chroma search test completed")
        except Exception as e:
            print(f"Error testing sentence Chroma: {e}")
    
    print("\nStep 8: Saving results and statistics...")
    
    memory_stats = {}
    if word_embeddings is not None:
        memory_stats['word_embeddings_memory_mb'] = word_embeddings.nbytes / (1024 * 1024)
    if sentence_embeddings is not None:
        memory_stats['sentence_embeddings_memory_mb'] = sentence_embeddings.nbytes / (1024 * 1024)
    
    total_memory = sum(memory_stats.values())
    memory_stats['total_embeddings_memory_mb'] = total_memory
    
    results = {
        'pdf_file': pdf_path,
        'model_used': model_name,
        'device_used': str(device),
        'processing_stats': {
            'word_chunks_count': len(word_chunks),
            'sentence_chunks_count': len(sentence_chunks),
            'word_chunks_stats': word_stats,
            'sentence_chunks_stats': sentence_stats
        },
        'embeddings_stats': {
            'word_embeddings_shape': word_embeddings.shape if word_embeddings is not None else None,
            'sentence_embeddings_shape': sentence_embeddings.shape if sentence_embeddings is not None else None,
            'embedding_dimension': word_embeddings.shape[1] if word_embeddings is not None else None,
            'memory_usage': memory_stats
        },
        'index_creation_time': {
            'faiss_time': faiss_time,
            'chroma_time': chroma_time
        },
        'test_results': {
            'test_query': test_query,
            'word_faiss_distances': word_distances[0].tolist() if word_distances is not None else None,
            'sentence_faiss_distances': sentence_distances[0].tolist() if sentence_distances is not None else None
        },
        'optimization_settings': {
            'embedding_batch_size': embedding_batch_size,
            'faiss_index_type': 'auto',
            'chroma_batch_size': 500
        },
        'success_flags': {
            'word_chunks_created': len(word_chunks) > 0,
            'sentence_chunks_created': len(sentence_chunks) > 0,
            'word_embeddings_generated': word_embeddings is not None,
            'sentence_embeddings_generated': sentence_embeddings is not None,
            'word_faiss_created': word_faiss_index is not None,
            'sentence_faiss_created': sentence_faiss_index is not None,
            'word_chroma_created': word_chroma is not None,
            'sentence_chroma_created': sentence_chroma is not None
        }
    }
    
    save_results(results, "phase3_pdf_processing_results.json")
    
    if word_embeddings is not None:
        del word_embeddings
    if sentence_embeddings is not None:
        del sentence_embeddings
    del word_chunks, sentence_chunks
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n" + "="*50)
    print("🎉 PHASE 3 COMPLETED!")
    print(f"✅ Device used: {device}")
    print(f"✅ Word chunks: {results['processing_stats']['word_chunks_count']} chunks")
    print(f"✅ Sentence chunks: {results['processing_stats']['sentence_chunks_count']} chunks")
    
    if total_memory > 0:
        print(f"✅ Total memory used: {total_memory:.1f} MB")
    
    success_flags = results['success_flags']
    print(f"✅ FAISS indexes: Word={success_flags['word_faiss_created']}, Sentence={success_flags['sentence_faiss_created']}")
    print(f"✅ Chroma collections: Word={success_flags['word_chroma_created']}, Sentence={success_flags['sentence_chroma_created']}")
    print(f"✅ All results saved to results/")
    
    print(f"\n📁 FILES CREATED:")
    if success_flags['word_chunks_created']:
        print(f"  - data/processed/drugs_word_chunks.csv")
    if success_flags['sentence_chunks_created']:
        print(f"  - data/processed/drugs_sentence_chunks.csv")
    if success_flags['word_faiss_created']:
        print(f"  - results/faiss/drugs_word_chunks.index")
    if success_flags['sentence_faiss_created']:
        print(f"  - results/faiss/drugs_sentence_chunks.index")
    if success_flags['word_chroma_created'] or success_flags['sentence_chroma_created']:
        print(f"  - chroma_db/ (Chroma database directory)")
    print(f"  - results/phase3_pdf_processing_results.json")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        if results:
            print("\n✅ Phase 3 executed successfully!")
        else:
            print("\n❌ Phase 3 failed!")
    except Exception as e:
        print(f"\n💥 Critical error in Phase 3: {e}")
        import traceback
        traceback.print_exc()