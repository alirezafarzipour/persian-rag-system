#!/usr/bin/env python3
"""
Debug script for RAG testing - RAG quick test and debug
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import RetrievalSystem
from src.llama_client import LlamaClient
from src.evaluation import RAGEvaluator
import pandas as pd

def test_retrieval_only():
    """test retrieval"""
    print("=== Testing Retrieval System ===")
    
    chunk_file = "data/processed/drugs_word_chunks.csv"
    faiss_index = "results/faiss/drugs_word_chunks.index"
    
    if not os.path.exists(chunk_file):
        print(f"✗ Chunk file not found: {chunk_file}")
        return False
    
    if not os.path.exists(faiss_index):
        print(f"✗ FAISS index not found: {faiss_index}")
        return False
    
    models_dir = "models"
    model_path = None
    
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            full_path = os.path.join(models_dir, item)
            if os.path.isdir(full_path) and "finetuned" in item:
                model_path = full_path
                break
    
    if not model_path:
        print("No fine-tuned model found, using base model...")
        model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    try:
        retriever = RetrievalSystem(
            method="dense",
            model_path=model_path,
            device="cpu" 
        )
        
        if not retriever.load_chunks_and_index(chunk_file, faiss_index):
            print("✗ Failed to load data")
            return False
        
        test_queries = [
            "دارو چیست؟",
            "عوارض جانبی دارو",
            "نحوه مصرف دارو",
            "تداخل دارویی چیست؟",
            "نگهداری دارو"
        ]
        
        print("\n--- Retrieval Test Results ---")
        for query in test_queries:
            print(f"\nQuery: {query}")
            contexts, metadata = retriever.get_contexts_for_rag(query, top_k=3)
            
            if contexts:
                print(f"Found {len(contexts)} contexts:")
                for i, (ctx, meta) in enumerate(zip(contexts, metadata)):
                    print(f"  {i+1}. Score: {meta['score']:.3f}")
                    print(f"     Text: {ctx[:100]}...")
            else:
                print("  No contexts found!")
        
        print("\n✓ Retrieval test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Retrieval test failed: {e}")
        return False

def test_llama_only():
    """تست فقط LLaMA connection"""
    print("\n=== Testing LLaMA Connection ===")
    
    try:
        client = LlamaClient("http://127.0.0.1:8080")
        
        test_prompts = [
            "سلام، چطوری؟",
            "دارو چیست؟",
            "بر اساس اطلاعات زیر سوال را پاسخ دهید:\nمتن: آسپرین دارویی ضد درد است.\nسوال: آسپرین چه نوع دارویی است؟\nپاسخ:"
        ]
        
        print("\n--- LLaMA Test Results ---")
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest {i+1}: {prompt[:50]}...")
            
            response = client.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.1
            )
            
            if response:
                print(f"Response: {response[:200]}...")
                print("✓ Success")
            else:
                print("✗ No response")
        
        print("\n✓ LLaMA test completed")
        return True
        
    except Exception as e:
        print(f"✗ LLaMA test failed: {e}")
        return False

def test_full_rag_pipeline():
    """تست کل pipeline RAG"""
    print("\n=== Testing Full RAG Pipeline ===")
    
    try:
        df = pd.read_csv("data/processed/test_data.csv", encoding='utf-8')
        test_questions = df.head(3).to_dict('records')  # فقط 3 سوال برای تست
    except Exception as e:
        print(f"Could not load test data: {e}")

        test_questions = [
            {"question": "دارو چیست؟", "answer": "دارو ماده‌ای است که برای درمان بیماری استفاده می‌شود"},
            {"question": "عوارض جانبی چیست؟", "answer": "اثرات ناخواسته دارو"},
            {"question": "نحوه نگهداری دارو", "answer": "در جای خنک و خشک نگهداری شود"}
        ]
    
    chunk_file = "data/processed/drugs_word_chunks.csv"
    faiss_index = "results/faiss/drugs_word_chunks.index"
    
    model_path = None
    models_dir = "models"
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            full_path = os.path.join(models_dir, item)
            if os.path.isdir(full_path) and "finetuned" in item:
                model_path = full_path
                break
    
    if not model_path:
        model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    try:
        retriever = RetrievalSystem(
            method="dense",
            model_path=model_path,
            device="cpu"
        )
        
        if not retriever.load_chunks_and_index(chunk_file, faiss_index):
            print("✗ Could not load retrieval data")
            return False
        
        # LLaMA client
        llama_client = LlamaClient("http://127.0.0.1:8080")
        
        # evaluator
        evaluator = RAGEvaluator("http://127.0.0.1:8080")
        
        print("\n--- Full Pipeline Test Results ---")
        
        for i, item in enumerate(test_questions):
            question = item['question']
            expected_answer = item['answer']
            
            print(f"\n=== Test {i+1} ===")
            print(f"Question: {question}")
            print(f"Expected: {expected_answer}")
            
            # Step 1: Retrieval
            contexts, metadata = retriever.get_contexts_for_rag(question, top_k=3)
            print(f"Retrieved {len(contexts)} contexts")
            
            if not contexts:
                print("✗ No contexts retrieved")
                continue
            
            if metadata:
                print(f"Best context (score: {metadata[0]['score']:.3f}):")
                print(f"  {contexts[0][:150]}...")
            
            # Step 2: Generation
            generated_answer = llama_client.answer_question(question, contexts)
            
            if generated_answer:
                print(f"Generated: {generated_answer}")
                
                # Step 3: Evaluation
                f1 = evaluator.f1_score(generated_answer, expected_answer)
                bleu = evaluator.bleu_score(generated_answer, expected_answer)
                
                print(f"F1 Score: {f1:.3f}")
                print(f"BLEU Score: {bleu:.3f}")
                print("✓ Pipeline completed successfully")
            else:
                print("✗ No answer generated")
        
        print("\n✓ Full RAG pipeline test completed")
        return True
        
    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_system_status():
    """Show System Status Check"""
    print("=== System Status Check ===")
    
    required_files = [
        "data/processed/test_data.csv",
        "data/processed/drugs_word_chunks.csv", 
        "data/processed/drugs_sentence_chunks.csv",
        "results/faiss/drugs_word_chunks.index",
        "results/faiss/drugs_sentence_chunks.index"
    ]
    
    print("\nRequired Files:")
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
    
    models_dir = "models"
    print(f"\nAvailable Models:")
    if os.path.exists(models_dir):
        found_models = 0
        for item in os.listdir(models_dir):
            full_path = os.path.join(models_dir, item)
            if os.path.isdir(full_path):
                print(f"  ✓ {item}")
                found_models += 1
        
        if found_models == 0:
            print("  ✗ No models found")
    else:
        print("  ✗ Models directory not found")
    
    print(f"\nLLaMA Server:")
    try:
        client = LlamaClient("http://127.0.0.1:8080")
        info = client.get_server_info()
        if info['status'] == 'connected':
            print(f"  ✓ Connected to {info['base_url']}")
            print(f"  ✓ Available endpoints: {len(info['endpoints'])}")
        else:
            print(f"  ✗ Not connected")
    except:
        print(f"  ✗ Connection failed")

def main():
    print("🔧 RAG Debug & Test Script")
    print("=" * 50)
    
    show_system_status()
    
    print("\nSelect test to run:")
    print("1. Test Retrieval Only")
    print("2. Test LLaMA Only") 
    print("3. Test Full RAG Pipeline")
    print("4. Run All Tests")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            test_retrieval_only()
        elif choice == "2":
            test_llama_only()
        elif choice == "3":
            test_full_rag_pipeline()
        elif choice == "4":
            print("\n" + "="*50)
            success1 = test_retrieval_only()
            print("\n" + "="*50)
            success2 = test_llama_only() 
            print("\n" + "="*50)
            success3 = test_full_rag_pipeline()
            
            print(f"\n🏁 All Tests Summary:")
            print(f"  Retrieval: {'✓' if success1 else '✗'}")
            print(f"  LLaMA: {'✓' if success2 else '✗'}")
            print(f"  Full Pipeline: {'✓' if success3 else '✗'}")
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()