from datasets import load_dataset
import PyPDF2
from hazm import Normalizer, WordTokenizer

class DataLoader:
    def __init__(self):
        self.normalizer = Normalizer()
        self.tokenizer = WordTokenizer()
    
    def load_datasets(self):
        """بارگذاری دیتاست‌ها"""
        pquad = load_dataset("Gholamreza/pquad")
        persian_qa = load_dataset("SajjadAyoubi/persian_qa") 
        return pquad, persian_qa
    
    def extract_pdf(self, pdf_path):
        """استخراج متن از PDF"""
        # پیاده‌سازی ساده
        pass
    
    def preprocess_text(self, text):
        """پیش‌پردازش متن فارسی"""
        return self.normalizer.normalize(text)