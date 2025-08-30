# #!/usr/bin/env python3
# """
# Phase 4: RAG Evaluation with Multiple Embedding Models
# Enhanced version with individual FAISS indices and improved LLaMA response cleaning
# """

# import sys
# import os
# import warnings
# import gc
# import time
# import re
# from typing import List, Dict, Any

# warnings.filterwarnings("ignore")
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.evaluation import RAGEvaluator
# from src.retrieval import RetrievalSystem
# from src.llama_client import LlamaClient
# from src.utils import load_config, save_results, ensure_directories
# import pandas as pd

# def clean_prediction(text):
#     """Clean model prediction from artifacts and extra text."""
#     if not text:
#         return ""
    
#     # Remove special tokens
#     text = re.sub(r'<\|[^|]*\|>', '', text)
#     text = re.sub(r'user[a-zA-Z]*', '', text)
#     text = re.sub(r'assistant[a-zA-Z]*', '', text)
#     text = re.sub(r'<[^>]*>', '', text)
    
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     # If prediction is too long, take first meaningful part
#     sentences = text.split('.')
#     if len(sentences) > 1:
#         # Take first sentence if it's not empty
#         first_sentence = sentences[0].strip()
#         if first_sentence:
#             text = first_sentence
    
#     # If still too long, take first 50 characters
#     if len(text) > 50:
#         text = text[:50].strip()
    
#     return text

# def check_prerequisites():
#     """Check prerequisites"""
#     print("Checking prerequisites...")
    
#     required_files = [
#         "data/processed/test_data.csv",
#         "data/processed/drugs_word_chunks.csv",
#         "data/processed/drugs_sentence_chunks.csv"
#     ]
    
#     missing_files = []
#     for file_path in required_files:
#         if not os.path.exists(file_path):
#             missing_files.append(file_path)
    
#     if missing_files:
#         print("✗ Missing required files:")
#         for file in missing_files:
#             print(f"  - {file}")
#         print("\nPlease run Phase 1, 2, and 3 first!")
#         return False
    
#     # Check for fine-tuned models
#     models_dir = "models"
#     trained_models = []
#     if os.path.exists(models_dir):
#         for item in os.listdir(models_dir):
#             model_path = os.path.join(models_dir, item)
#             if os.path.isdir(model_path) and "finetuned" in item:
#                 trained_models.append(model_path)
    
#     if not trained_models:
#         print("⚠ No fine-tuned models found, will use base models")
#     else:
#         print(f"✓ Found {len(trained_models)} fine-tuned models")
    
#     # Check FAISS directory
#     faiss_dir = "results/faiss"
#     if not os.path.exists(faiss_dir):
#         print("✗ FAISS indexes not found. Please run Phase 3!")
#         return False
    
#     print("✓ All prerequisites check passed")
#     return True

# def load_test_data(sample_size: int = None) -> List[Dict]:
#     """Load test data"""
#     print("Loading test data...")
    
#     try:
#         df = pd.read_csv("data/processed/test_data.csv", encoding='utf-8')
#         test_data = df.to_dict('records')
        
#         if sample_size and len(test_data) > sample_size:
#             test_data = test_data[:sample_size]
#             print(f"Using sample of {sample_size} questions from {len(df)} total")
#         else:
#             print(f"Loaded {len(test_data)} test questions")
        
#         return test_data
        
#     except Exception as e:
#         print(f"Error loading test data: {e}")
#         return []

# def find_models_and_indices():
#     """Find models and their corresponding FAISS indices"""
#     models_dir = "models"
#     faiss_dir = "results/faiss"
    
#     model_paths = []
#     model_indices = {}  # {model_name: {chunk_type: index_file}}
    
#     # First check fine-tuned models
#     if os.path.exists(models_dir):
#         for item in os.listdir(models_dir):
#             model_path = os.path.join(models_dir, item)
#             if os.path.isdir(model_path) and "finetuned" in item:
#                 model_paths.append(model_path)
#                 model_name = os.path.basename(model_path)
                
#                 # Find corresponding FAISS indices
#                 model_indices[model_name] = {}
#                 for chunk_type in ["word", "sentence"]:
#                     # Look for model-specific index
#                     index_file = f"{faiss_dir}/{model_name}_drugs_{chunk_type}_chunks.index"
#                     if os.path.exists(index_file):
#                         model_indices[model_name][chunk_type] = index_file
#                     else:
#                         # Fall back to generic index if model-specific doesn't exist
#                         generic_index = f"{faiss_dir}/drugs_{chunk_type}_chunks.index"
#                         if os.path.exists(generic_index):
#                             model_indices[model_name][chunk_type] = generic_index
    
#     # If no fine-tuned models, use base models
#     if not model_paths:
#         print("No fine-tuned models found, adding base models...")
#         base_models = [
#             "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#             "sentence-transformers/distiluse-base-multilingual-cased-v2",
#             "intfloat/multilingual-e5-base"
#         ]
        
#         for base_model in base_models:
#             model_name = base_model.split('/')[-1]
            
#             # Check if we have indices for this base model
#             has_index = False
#             model_indices[model_name] = {}
            
#             for chunk_type in ["word", "sentence"]:
#                 # Look for model-specific index first
#                 index_file = f"{faiss_dir}/{model_name}_drugs_{chunk_type}_chunks.index"
#                 if os.path.exists(index_file):
#                     model_indices[model_name][chunk_type] = index_file
#                     has_index = True
#                 else:
#                     # Fall back to generic index
#                     generic_index = f"{faiss_dir}/drugs_{chunk_type}_chunks.index"
#                     if os.path.exists(generic_index):
#                         model_indices[model_name][chunk_type] = generic_index
#                         has_index = True
            
#             if has_index:
#                 model_paths.append(base_model)
#             else:
#                 print(f"⚠ No FAISS indices found for {model_name}")
    
#     return model_paths, model_indices

# def create_missing_embeddings_and_indices(model_paths: List[str], model_indices: Dict):
#     """Create missing embeddings and FAISS indices for models"""
#     from sentence_transformers import SentenceTransformer
#     import faiss
    
#     print("\n=== Creating Missing Embeddings and Indices ===")
    
#     chunk_files = {
#         "word": "data/processed/drugs_word_chunks.csv",
#         "sentence": "data/processed/drugs_sentence_chunks.csv"
#     }
    
#     for model_path in model_paths:
#         model_name = os.path.basename(model_path)
#         print(f"\nProcessing model: {model_name}")
        
#         # Check if we need to create any indices for this model
#         needs_creation = False
#         for chunk_type in ["word", "sentence"]:
#             model_specific_index = f"results/faiss/{model_name}_drugs_{chunk_type}_chunks.index"
#             if not os.path.exists(model_specific_index):
#                 needs_creation = True
#                 break
        
#         if not needs_creation:
#             print(f"✓ All indices exist for {model_name}")
#             continue
        
#         # Load the embedding model
#         print(f"Loading embedding model: {model_path}")
#         try:
#             device = "cuda" if os.path.exists("/usr/local/cuda") else "cpu"
#             embedding_model = SentenceTransformer(model_path, device=device)
#         except Exception as e:
#             print(f"✗ Failed to load model {model_path}: {e}")
#             continue
        
#         # Process each chunk type
#         for chunk_type in ["word", "sentence"]:
#             chunk_file = chunk_files[chunk_type]
#             model_specific_index = f"results/faiss/{model_name}_drugs_{chunk_type}_chunks.index"
            
#             if os.path.exists(model_specific_index):
#                 print(f"✓ Index already exists: {model_specific_index}")
#                 # Update model_indices to point to the model-specific index
#                 if model_name not in model_indices:
#                     model_indices[model_name] = {}
#                 model_indices[model_name][chunk_type] = model_specific_index
#                 continue
            
#             if not os.path.exists(chunk_file):
#                 print(f"⚠ Chunk file not found: {chunk_file}")
#                 continue
            
#             print(f"Creating {chunk_type} embeddings for {model_name}...")
            
#             try:
#                 # Load chunks
#                 df = pd.read_csv(chunk_file, encoding='utf-8')
#                 texts = df['text'].tolist()
                
#                 print(f"  Encoding {len(texts)} chunks...")
                
#                 # Create embeddings in batches
#                 batch_size = 32
#                 embeddings = []
                
#                 for i in range(0, len(texts), batch_size):
#                     batch_texts = texts[i:i + batch_size]
#                     batch_embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)
#                     embeddings.extend(batch_embeddings)
                    
#                     if i % (batch_size * 10) == 0:
#                         print(f"    Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
#                         gc.collect()
                
#                 embeddings = np.array(embeddings).astype('float32')
#                 print(f"  Created embeddings shape: {embeddings.shape}")
                
#                 # Create FAISS index
#                 print(f"  Creating FAISS index...")
#                 dimension = embeddings.shape[1]
#                 index = faiss.IndexFlatL2(dimension)
#                 index.add(embeddings)
                
#                 # Save model-specific index
#                 os.makedirs("results/faiss", exist_ok=True)
#                 faiss.write_index(index, model_specific_index)
#                 print(f"✓ Saved FAISS index: {model_specific_index}")
                
#                 # Update model_indices
#                 if model_name not in model_indices:
#                     model_indices[model_name] = {}
#                 model_indices[model_name][chunk_type] = model_specific_index
                
#                 # Clean up
#                 del embeddings, index
#                 gc.collect()
                
#             except Exception as e:
#                 print(f"✗ Error creating {chunk_type} embeddings for {model_name}: {e}")
        
#         # Clean up model
#         del embedding_model
#         gc.collect()
#         if hasattr(torch, 'cuda') and torch.cuda.is_available():
#             torch.cuda.empty_cache()

# def test_llama_connection(llama_url: str = "http://127.0.0.1:8080") -> bool:
#     """Test connection to LLaMA server"""
#     print(f"Testing connection to LLaMA server at {llama_url}...")
    
#     try:
#         client = LlamaClient(llama_url)
#         test_response = client.generate("سلام", max_tokens=10)
        
#         if test_response:
#             cleaned_response = clean_prediction(test_response)
#             print("✓ LLaMA server is responding correctly")
#             print(f"  Test response: {cleaned_response[:50]}...")
#             return True
#         else:
#             print("⚠ LLaMA server is running but not responding properly")
#             return False
            
#     except Exception as e:
#         print(f"✗ Could not connect to LLaMA server: {e}")
#         print("  Please make sure the server is running on 127.0.0.1:8080")
#         return False

# def run_single_model_evaluation(model_path: str, chunk_file: str, faiss_index: str,
#                                test_data: List[Dict], evaluator: RAGEvaluator) -> Dict[str, Any]:
#     """Single embedding model evaluation in RAG"""
#     model_name = os.path.basename(model_path)
#     print(f"\n=== Evaluating {model_name} ===")
    
#     try:
#         # Create retriever with specific model and index
#         retriever = RetrievalSystem(
#             method="dense",  # Use dense retrieval for embedding comparison
#             model_path=model_path,
#             device="cuda" if os.path.exists("/usr/local/cuda") else "cpu"
#         )
        
#         if not retriever.load_chunks_and_index(chunk_file, faiss_index):
#             print(f"✗ Failed to load data for {model_name}")
#             return {}
        
#         # Evaluate with enhanced response cleaning
#         results = evaluator.evaluate_single_rag(
#             retriever=retriever,
#             test_data=test_data,
#             model_name=model_name,
#             sample_size=None
#         )
        
#         # Clean up
#         retriever.cleanup()
#         del retriever
#         gc.collect()
        
#         return results
        
#     except Exception as e:
#         print(f"✗ Error evaluating {model_name}: {e}")
#         return {}

# def main():
#     print("=== PHASE 4: Enhanced RAG Evaluation with Multiple Models ===\n")
    
#     if not check_prerequisites():
#         return
    
#     ensure_directories()
#     config = load_config()
    
#     # Test LLaMA connection
#     llama_url = "http://127.0.0.1:8080"
#     if not test_llama_connection(llama_url):
#         print("\nCannot proceed without LLaMA server. Please:")
#         print("1. Make sure LLaMA server is running")
#         print("2. Check that it's accessible at 127.0.0.1:8080")
#         print("3. Test with a simple request first")
#         return

#     # Load test data
#     sample_size = config.get('evaluation', {}).get('test_size', 100)
#     if isinstance(sample_size, float):
#         sample_size = int(sample_size * 1000)
    
#     sample_size = min(sample_size, 200)  # Max 200 questions for test
    
#     test_data = load_test_data(sample_size)
#     if not test_data:
#         print("Could not load test data!")
#         return
    
#     # Find models and indices
#     model_paths, model_indices = find_models_and_indices()
    
#     if not model_paths:
#         print("No models found for evaluation!")
#         return
    
#     # Create missing embeddings and indices
#     create_missing_embeddings_and_indices(model_paths, model_indices)
    
#     print(f"\nFound {len(model_paths)} models to evaluate:")
#     for i, model_path in enumerate(model_paths):
#         model_name = os.path.basename(model_path)
#         indices_info = model_indices.get(model_name, {})
#         print(f"  {i+1}. {model_name}")
#         for chunk_type, index_file in indices_info.items():
#             print(f"     {chunk_type}: {index_file}")
    
#     # Determine available chunk types
#     chunk_types = []
#     if os.path.exists("data/processed/drugs_word_chunks.csv"):
#         chunk_types.append(("word", "data/processed/drugs_word_chunks.csv"))
#     if os.path.exists("data/processed/drugs_sentence_chunks.csv"):
#         chunk_types.append(("sentence", "data/processed/drugs_sentence_chunks.csv"))
    
#     if not chunk_types:
#         print("No chunk files found!")
#         return
    
#     # Create enhanced evaluator
#     evaluator = RAGEvaluator(llama_url)
    
#     all_results = {}
    
#     # Evaluate each chunk type
#     for chunk_type, chunk_file in chunk_types:
#         print(f"\n{'='*60}")
#         print(f"EVALUATING WITH {chunk_type.upper()} CHUNKS")
#         print('='*60)
        
#         chunk_results = {}
        
#         for i, model_path in enumerate(model_paths):
#             model_name = os.path.basename(model_path)
            
#             print(f"\nModel {i+1}/{len(model_paths)}: {model_name}")
#             print("-" * 50)
            
#             # Get model-specific index for this chunk type
#             if model_name not in model_indices or chunk_type not in model_indices[model_name]:
#                 print(f"✗ No {chunk_type} index found for {model_name}")
#                 continue
            
#             index_file = model_indices[model_name][chunk_type]
            
#             if not os.path.exists(index_file):
#                 print(f"✗ Index file not found: {index_file}")
#                 continue
            
#             # Run evaluation
#             model_results = run_single_model_evaluation(
#                 model_path, chunk_file, index_file, test_data, evaluator
#             )
            
#             if model_results:
#                 chunk_results.update(model_results)
                
#                 print(f"\n📊 {model_name} Results Summary:")
#                 key_metrics = ['f1_score', 'bleu_score', 'exact_match', 'success_rate', 'total_time']
#                 for metric in key_metrics:
#                     key = f"{model_name}_{metric}"
#                     if key in model_results:
#                         print(f"  {metric}: {model_results[key]:.4f}")
            
#             time.sleep(2)  # Brief pause between models
        
#         # Store results for this chunk type
#         all_results[f"{chunk_type}_chunks"] = chunk_results
        
#         # Analyze comparison for this chunk type
#         if len(chunk_results) > 0:
#             model_performances = {}
#             for model_path in model_paths:
#                 model_name = os.path.basename(model_path)
#                 model_data = {k: v for k, v in chunk_results.items() if k.startswith(model_name)}
#                 if model_data:
#                     model_performances[model_name] = model_data
            
#             if model_performances:
#                 comparison = evaluator._analyze_model_comparison(model_performances)
#                 all_results[f"{chunk_type}_chunks_comparison"] = comparison
    
#     # Final results summary
#     print(f"\n{'='*60}")
#     print("FINAL RESULTS SUMMARY")
#     print('='*60)
    
#     for chunk_type in ["word", "sentence"]:
#         comparison_key = f"{chunk_type}_chunks_comparison"
#         if comparison_key in all_results:
#             comparison = all_results[comparison_key]
            
#             print(f"\n🏆 Best Models for {chunk_type} chunks:")
#             if 'best_models' in comparison:
#                 for metric in ['f1_score', 'bleu_score', 'success_rate']:
#                     if metric in comparison['best_models']:
#                         info = comparison['best_models'][metric]
#                         print(f"  {metric}: {info['model']} ({info['score']:.4f})")
    
#     # Save results
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     results_file = f"phase4_enhanced_rag_evaluation_{timestamp}.json"
    
#     all_results['evaluation_metadata'] = {
#         'timestamp': timestamp,
#         'llama_url': llama_url,
#         'num_test_questions': len(test_data),
#         'models_evaluated': [os.path.basename(path) for path in model_paths],
#         'chunk_types': [ct[0] for ct in chunk_types],
#         'sample_size': sample_size,
#         'enhancement': 'individual_faiss_indices_per_model'
#     }
    
#     evaluator.save_evaluation_results(all_results, results_file)
    
#     # Create report
#     report = evaluator.create_evaluation_report(all_results)
#     report_file = f"results/phase4_enhanced_rag_report_{timestamp}.md"
    
#     with open(report_file, 'w', encoding='utf-8') as f:
#         f.write(report)
    
#     print(f"\n✓ Phase 4 Enhanced completed successfully!")
#     print(f"📁 Results saved to:")
#     print(f"  - {results_file}")
#     print(f"  - {report_file}")
    
#     print(f"\n📈 Performance Summary:")
#     print(f"  Total models evaluated: {len(model_paths)}")
#     print(f"  Total questions processed: {len(test_data)}")
#     print(f"  Chunk types tested: {len(chunk_types)}")
    
#     if chunk_types:
#         for chunk_type, _ in chunk_types:
#             comparison_key = f"{chunk_type}_chunks_comparison"
#             if comparison_key in all_results and 'best_models' in all_results[comparison_key]:
#                 best_f1_info = all_results[comparison_key]['best_models'].get('f1_score')
#                 if best_f1_info:
#                     print(f"  Best {chunk_type} F1: {best_f1_info['model']} ({best_f1_info['score']:.4f})")
    
#     return all_results

# if __name__ == "__main__":
#     try:
#         results = main()
#         print("\n🎉 Enhanced RAG evaluation completed successfully!")
#     except KeyboardInterrupt:
#         print("\n⚠ Evaluation interrupted by user")
#     except Exception as e:
#         print(f"\n✗ Evaluation failed: {e}")
#         import traceback
#         traceback.print_exc()

#!/usr/bin/env python3
"""
Phase 4: RAG Evaluation with Multiple Embedding Models
Enhanced version with individual FAISS indices and improved LLaMA response cleaning
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
    
    # Check for fine-tuned models
    models_dir = "models"
    trained_models = []
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path) and "finetuned" in item:
                trained_models.append(model_path)
    
    if not trained_models:
        print("⚠ No fine-tuned models found, will use base models")
    else:
        print(f"✓ Found {len(trained_models)} fine-tuned models")
    
    # Check FAISS directory
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
    """Find models and their corresponding FAISS indices"""
    models_dir = "models"
    faiss_dir = "results/faiss"
    
    model_paths = []
    model_indices = {}  # {model_name: {chunk_type: index_file}}
    
    # First check fine-tuned models
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path) and "finetuned" in item:
                model_paths.append(model_path)
                model_name = os.path.basename(model_path)
                
                # Find corresponding FAISS indices
                model_indices[model_name] = {}
                for chunk_type in ["word", "sentence"]:
                    # Look for model-specific index
                    index_file = f"{faiss_dir}/{model_name}_drugs_{chunk_type}_chunks.index"
                    if os.path.exists(index_file):
                        model_indices[model_name][chunk_type] = index_file
                    else:
                        # Fall back to generic index if model-specific doesn't exist
                        generic_index = f"{faiss_dir}/drugs_{chunk_type}_chunks.index"
                        if os.path.exists(generic_index):
                            model_indices[model_name][chunk_type] = generic_index
    
    # If no fine-tuned models, use base models
    if not model_paths:
        print("No fine-tuned models found, adding base models...")
        base_models = [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/distiluse-base-multilingual-cased-v2",
            "intfloat/multilingual-e5-base"
        ]
        
        for base_model in base_models:
            model_name = base_model.split('/')[-1]
            
            # Check if we have indices for this base model
            has_index = False
            model_indices[model_name] = {}
            
            for chunk_type in ["word", "sentence"]:
                # Look for model-specific index first
                index_file = f"{faiss_dir}/{model_name}_drugs_{chunk_type}_chunks.index"
                if os.path.exists(index_file):
                    model_indices[model_name][chunk_type] = index_file
                    has_index = True
                else:
                    # Fall back to generic index
                    generic_index = f"{faiss_dir}/drugs_{chunk_type}_chunks.index"
                    if os.path.exists(generic_index):
                        model_indices[model_name][chunk_type] = generic_index
                        has_index = True
            
            if has_index:
                model_paths.append(base_model)
            else:
                print(f"⚠ No FAISS indices found for {model_name}")
    
    return model_paths, model_indices

def create_missing_embeddings_and_indices(model_paths: List[str], model_indices: Dict):
    """Create missing embeddings and FAISS indices for models"""
    from sentence_transformers import SentenceTransformer
    import faiss
    
    print("\n=== Creating Missing Embeddings and Indices ===")
    
    chunk_files = {
        "word": "data/processed/drugs_word_chunks.csv",
        "sentence": "data/processed/drugs_sentence_chunks.csv"
    }
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"\nProcessing model: {model_name}")
        
        # Check if we need to create any indices for this model
        needs_creation = False
        for chunk_type in ["word", "sentence"]:
            model_specific_index = f"results/faiss/{model_name}_drugs_{chunk_type}_chunks.index"
            if not os.path.exists(model_specific_index):
                needs_creation = True
                break
        
        if not needs_creation:
            print(f"✓ All indices exist for {model_name}")
            continue
        
        # Load the embedding model
        print(f"Loading embedding model: {model_path}")
        try:
            device = "cuda" if os.path.exists("/usr/local/cuda") else "cpu"
            embedding_model = SentenceTransformer(model_path, device=device)
        except Exception as e:
            print(f"✗ Failed to load model {model_path}: {e}")
            continue
        
        # Process each chunk type
        for chunk_type in ["word", "sentence"]:
            chunk_file = chunk_files[chunk_type]
            model_specific_index = f"results/faiss/{model_name}_drugs_{chunk_type}_chunks.index"
            
            if os.path.exists(model_specific_index):
                print(f"✓ Index already exists: {model_specific_index}")
                # Update model_indices to point to the model-specific index
                if model_name not in model_indices:
                    model_indices[model_name] = {}
                model_indices[model_name][chunk_type] = model_specific_index
                continue
            
            if not os.path.exists(chunk_file):
                print(f"⚠ Chunk file not found: {chunk_file}")
                continue
            
            print(f"Creating {chunk_type} embeddings for {model_name}...")
            
            try:
                # Load chunks
                df = pd.read_csv(chunk_file, encoding='utf-8')
                texts = df['text'].tolist()
                
                print(f"  Encoding {len(texts)} chunks...")
                
                # Create embeddings in batches
                batch_size = 32
                embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)
                    embeddings.extend(batch_embeddings)
                    
                    if i % (batch_size * 10) == 0:
                        print(f"    Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
                        gc.collect()
                
                embeddings = np.array(embeddings).astype('float32')
                print(f"  Created embeddings shape: {embeddings.shape}")
                
                # Create FAISS index
                print(f"  Creating FAISS index...")
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                
                # Save model-specific index
                os.makedirs("results/faiss", exist_ok=True)
                faiss.write_index(index, model_specific_index)
                print(f"✓ Saved FAISS index: {model_specific_index}")
                
                # Update model_indices
                if model_name not in model_indices:
                    model_indices[model_name] = {}
                model_indices[model_name][chunk_type] = model_specific_index
                
                # Clean up
                del embeddings, index
                gc.collect()
                
            except Exception as e:
                print(f"✗ Error creating {chunk_type} embeddings for {model_name}: {e}")
        
        # Clean up model
        del embedding_model
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

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

def run_single_model_evaluation(model_path: str, chunk_file: str, faiss_index: str,
                               test_data: List[Dict], evaluator: RAGEvaluator) -> Dict[str, Any]:
    """Single embedding model evaluation in RAG"""
    model_name = os.path.basename(model_path)
    print(f"\n=== Evaluating {model_name} ===")
    
    try:
        # Create retriever with specific model and index
        retriever = RetrievalSystem(
            method="dense",  # Use dense retrieval for embedding comparison
            model_path=model_path,
            device="cuda" if os.path.exists("/usr/local/cuda") else "cpu"
        )
        
        if not retriever.load_chunks_and_index(chunk_file, faiss_index):
            print(f"✗ Failed to load data for {model_name}")
            return {}
        
        # Evaluate with enhanced response cleaning
        results = evaluator.evaluate_single_rag(
            retriever=retriever,
            test_data=test_data,
            model_name=model_name,
            sample_size=None
        )
        
        # Clean up
        retriever.cleanup()
        del retriever
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"✗ Error evaluating {model_name}: {e}")
        return {}

def main():
    print("=== PHASE 4: Enhanced RAG Evaluation with Multiple Models ===\n")
    
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
    sample_size = config.get('evaluation', {}).get('test_size', 100)
    if isinstance(sample_size, float):
        sample_size = int(sample_size * 1000)
    
    sample_size = min(sample_size, 200)  # Max 200 questions for test
    
    test_data = load_test_data(sample_size)
    if not test_data:
        print("Could not load test data!")
        return
    
    # Find models and indices
    model_paths, model_indices = find_models_and_indices()
    
    if not model_paths:
        print("No models found for evaluation!")
        return
    
    # Create missing embeddings and indices
    create_missing_embeddings_and_indices(model_paths, model_indices)
    
    print(f"\nFound {len(model_paths)} models to evaluate:")
    for i, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path)
        indices_info = model_indices.get(model_name, {})
        print(f"  {i+1}. {model_name}")
        for chunk_type, index_file in indices_info.items():
            print(f"     {chunk_type}: {index_file}")
    
    # Determine available chunk types
    chunk_types = []
    if os.path.exists("data/processed/drugs_word_chunks.csv"):
        chunk_types.append(("word", "data/processed/drugs_word_chunks.csv"))
    if os.path.exists("data/processed/drugs_sentence_chunks.csv"):
        chunk_types.append(("sentence", "data/processed/drugs_sentence_chunks.csv"))
    
    if not chunk_types:
        print("No chunk files found!")
        return
    
    # Create enhanced evaluator
    evaluator = RAGEvaluator(llama_url)
    
    all_results = {}
    
    # Evaluate each chunk type
    for chunk_type, chunk_file in chunk_types:
        print(f"\n{'='*60}")
        print(f"EVALUATING WITH {chunk_type.upper()} CHUNKS")
        print('='*60)
        
        chunk_results = {}
        
        for i, model_path in enumerate(model_paths):
            model_name = os.path.basename(model_path)
            
            print(f"\nModel {i+1}/{len(model_paths)}: {model_name}")
            print("-" * 50)
            
            # Get model-specific index for this chunk type
            if model_name not in model_indices or chunk_type not in model_indices[model_name]:
                print(f"✗ No {chunk_type} index found for {model_name}")
                continue
            
            index_file = model_indices[model_name][chunk_type]
            
            if not os.path.exists(index_file):
                print(f"✗ Index file not found: {index_file}")
                continue
            
            # Run evaluation
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
            
            time.sleep(2)  # Brief pause between models
        
        # Store results for this chunk type
        all_results[f"{chunk_type}_chunks"] = chunk_results
        
        # Analyze comparison for this chunk type
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
    
    # Final results summary
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
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"phase4_enhanced_rag_evaluation_{timestamp}.json"
    
    all_results['evaluation_metadata'] = {
        'timestamp': timestamp,
        'llama_url': llama_url,
        'num_test_questions': len(test_data),
        'models_evaluated': [os.path.basename(path) for path in model_paths],
        'chunk_types': [ct[0] for ct in chunk_types],
        'sample_size': sample_size,
        'enhancement': 'individual_faiss_indices_per_model'
    }
    
    evaluator.save_evaluation_results(all_results, results_file)
    
    # Create report
    report = evaluator.create_evaluation_report(all_results)
    report_file = f"results/phase4_enhanced_rag_report_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Phase 4 Enhanced completed successfully!")
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