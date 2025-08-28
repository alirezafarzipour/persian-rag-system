from datasets import load_dataset
import PyPDF2
import pdfplumber
from hazm import Normalizer, WordTokenizer
from .utils import PersianTextProcessor
import pandas as pd
from typing import List, Dict, Tuple
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

class DataLoader:
    def __init__(self):
        self.text_processor = PersianTextProcessor()
        self.normalizer = Normalizer()
        self.tokenizer = WordTokenizer()
    
    def load_datasets(self) -> Tuple[Dict, Dict]:
        """loading datasets"""
        print("Loading Persian QA datasets...")
        
        try:
            # Gholamreza/pquad
            print("Loading pquad dataset...")
            pquad = load_dataset("Gholamreza/pquad", trust_remote_code=True)
            
            # SajjadAyoubi/persian_qa
            print("Loading persian_qa dataset...")
            persian_qa = load_dataset("SajjadAyoubi/persian_qa")
            
            print(f"PQuad loaded: {len(pquad['train'])} train, {len(pquad.get('validation', []))} val")
            print(f"Persian QA loaded: {len(persian_qa['train'])} samples")
            
            return pquad, persian_qa
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return None, None
    
    def extract_pdf(self, pdf_path: str) -> str:
        """Extracting text from PDF"""
        print(f"Extracting text from {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
        except Exception as e:
            print(f"pdfplumber failed: {e}, trying PyPDF2...")
            
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                print(f"PyPDF2 also failed: {e2}")
                raise e2
        
        # preprocess extract data
        processed_text = self.text_processor.normalize_text(text)
        
        print(f"Extracted {len(processed_text)} characters from PDF")
        return processed_text
    
    def preprocess_text(self, text: str) -> str:
        """persian text preprocessing"""
        return self.text_processor.normalize_text(text)
    
    def prepare_qa_data_for_training(self, pquad, persian_qa=None) -> List[Dict]:
        """prepare qa data for training"""
        print("Preparing QA data for training...")
        
        training_data = []
        
        # Processing pquad
        if pquad and 'train' in pquad:
            for item in pquad['train']:
                # Extracting questions, context, and answers
                question = self.preprocess_text(item.get('question', ''))
                context = self.preprocess_text(item.get('context', ''))
                answers = item.get('answers', {})
                
                if answers and 'text' in answers and len(answers['text']) > 0:
                    answer = self.preprocess_text(answers['text'][0])
                    
                    if len(question) > 10 and len(answer) > 5:
                        training_data.append({
                            'question': question,
                            'context': context,
                            'answer': answer,
                            'source': 'pquad'
                        })
        
        # Processing persian_qa
        if persian_qa and 'train' in persian_qa:
            for item in persian_qa['train']:
                question = self.preprocess_text(item.get('question', ''))
                answer = self.preprocess_text(item.get('answer', ''))
                
                if len(question) > 10 and len(answer) > 5:
                    training_data.append({
                        'question': question,
                        'context': '',
                        'answer': answer,
                        'source': 'persian_qa'
                    })
        
        print(f"Prepared {len(training_data)} QA pairs for training")
        return training_data
    
    def create_test_split(self, qa_data: List[Dict], test_size: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
        """Split data into train and test"""
        import random
        random.shuffle(qa_data)
        
        split_idx = int(len(qa_data) * (1 - test_size))
        train_data = qa_data[:split_idx]
        test_data = qa_data[split_idx:]
        
        print(f"Split: {len(train_data)} train, {len(test_data)} test")
        return train_data, test_data
    
    def save_processed_data(self, data: List[Dict], filename: str):
        """Save processed data"""
        filepath = f"data/processed/{filename}"
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Processed data saved to {filepath}")