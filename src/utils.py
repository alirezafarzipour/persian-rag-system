"""
Persian text processing utilities
"""
import re
from hazm import Normalizer, WordTokenizer, SentenceTokenizer
from typing import List, Dict, Any
import os

class PersianTextProcessor:
    def __init__(self):
        self.normalizer = Normalizer()
        self.word_tokenizer = WordTokenizer()
        self.sentence_tokenizer = SentenceTokenizer()
    
    def normalize_text(self, text: str) -> str:
        """Persian text normalization"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        text = text.strip()
        
        text = self.normalizer.normalize(text)
        
        return text
    
    def tokenize_words(self, text: str) -> List[str]:
        """tokenize to word"""
        normalized_text = self.normalize_text(text)
        return self.word_tokenizer.tokenize(normalized_text)
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """tokenize to sentence"""
        normalized_text = self.normalize_text(text)
        sentences = self.sentence_tokenizer.tokenize(normalized_text)
        return [sent.strip() for sent in sentences if sent.strip()]

def ensure_directories():
    """create directories"""
    directories = [
        'data/raw',
        'data/processed', 
        'results',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_results(results: Dict[str, Any], filename: str):
    """save results"""
    import json
    import pandas as pd
    
    filepath = f"results/{filename}"
    
    if filename.endswith('.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    elif filename.endswith('.csv'):
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False, encoding='utf-8')
    
    print(f"Results saved to {filepath}")

def load_config():
    """load config"""
    import yaml
    
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config