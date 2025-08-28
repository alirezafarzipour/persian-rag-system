#!/usr/bin/env python3
"""اسکریپت اصلی برای اجرای کل پروژه"""

import yaml
from src.data_loader import DataLoader
from src.chunking import TextChunker
from src.embedding_trainer import EmbeddingTrainer
from src.retrieval import RetrievalSystem
from src.evaluation import Evaluator

def main():
    # بارگذاری تنظیمات
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. بارگذاری داده‌ها
    print("Loading data...")
    data_loader = DataLoader()
    
    # 2. Chunking
    print("Chunking documents...")
    chunker = TextChunker(config)
    
    # 3. Fine-tuning مدل‌ها
    print("Training embedding models...")
    for model_name in config['models']:
        trainer = EmbeddingTrainer(model_name)
        # fine-tune و save
    
    # 4. ارزیابی
    print("Evaluating systems...")
    evaluator = Evaluator()
    
    # 5. مقایسه نتایج
    print("Comparing results...")

if __name__ == "__main__":
    main()