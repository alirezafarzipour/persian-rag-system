from .utils import PersianTextProcessor
from typing import List, Dict, Tuple
import math

class TextChunker:
    def __init__(self, config):
        self.config = config
        self.text_processor = PersianTextProcessor()
        
    def word_based_chunking(self, text: str) -> List[Dict[str, any]]:
        """Word-based chunking"""
        print("Performing word-based chunking...")
        
        chunk_size = self.config['chunking']['word_chunk_size']
        overlap = self.config['chunking']['word_overlap']
        
        normalized_text = self.text_processor.normalize_text(text)
        
        words = self.text_processor.tokenize_words(normalized_text)
        
        chunks = []
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + chunk_size, len(words))
            
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            chunk_info = {
                'id': f'word_chunk_{chunk_id}',
                'text': chunk_text,
                'start_word': start_idx,
                'end_word': end_idx,
                'num_words': len(chunk_words),
                'chunk_type': 'word_based',
                'overlap_words': overlap if start_idx > 0 else 0
            }
            
            chunks.append(chunk_info)
            
            start_idx = end_idx - overlap
            chunk_id += 1
            
            if start_idx >= end_idx:
                break
        
        print(f"✓ Created {len(chunks)} word-based chunks")
        return chunks
    
    def sentence_based_chunking(self, text: str) -> List[Dict[str, any]]:
        """Sentence-based chunking"""
        print("Performing sentence-based chunking...")
        
        sentences_per_chunk = self.config['chunking']['sentences_per_chunk']
        
        normalized_text = self.text_processor.normalize_text(text)
        
        sentences = self.text_processor.tokenize_sentences(normalized_text)
        
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = ' '.join(chunk_sentences)
            
            words_in_chunk = len(chunk_text.split())
            
            # ایجاد metadata
            chunk_info = {
                'id': f'sentence_chunk_{chunk_id}',
                'text': chunk_text,
                'start_sentence': i,
                'end_sentence': min(i + sentences_per_chunk, len(sentences)),
                'num_sentences': len(chunk_sentences),
                'num_words': words_in_chunk,
                'chunk_type': 'sentence_based'
            }
            
            chunks.append(chunk_info)
            chunk_id += 1
        
        print(f"✓ Created {len(chunks)} sentence-based chunks")
        return chunks
    
    def process_pdf_document(self, pdf_text: str) -> Tuple[List[Dict], List[Dict]]:
        """Full PDF processing with both chunking methods"""
        print("Processing PDF document with both chunking methods...")
        
        if not pdf_text or len(pdf_text.strip()) < 100:
            print("Warning: PDF text is too short or empty")
            return [], []
        
        # Word-based chunking
        word_chunks = self.word_based_chunking(pdf_text)
        
        # Sentence-based chunking  
        sentence_chunks = self.sentence_based_chunking(pdf_text)
        
        total_words = len(pdf_text.split())
        total_sentences = len(self.text_processor.tokenize_sentences(pdf_text))
        
        print(f"✓ PDF processing completed:")
        print(f"  - Total words: {total_words}")
        print(f"  - Total sentences: {total_sentences}")
        print(f"  - Word-based chunks: {len(word_chunks)}")
        print(f"  - Sentence-based chunks: {len(sentence_chunks)}")
        
        return word_chunks, sentence_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict[str, any]:
        """Chunks statistics"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk['text'].split()) for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'avg_words_per_chunk': sum(chunk_lengths) / len(chunk_lengths),
            'min_words_per_chunk': min(chunk_lengths),
            'max_words_per_chunk': max(chunk_lengths),
            'total_words': sum(chunk_lengths),
            'chunk_type': chunks[0]['chunk_type']
        }
        
        return stats
    
    def save_chunks(self, chunks: List[Dict], filename: str):
        """"Store chunks to a CSV file"""
        import pandas as pd
        import os
        
        os.makedirs('data/processed', exist_ok=True)
        filepath = f"data/processed/{filename}"
        
        df = pd.DataFrame(chunks)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"✓ Chunks saved to {filepath}")
    
    def load_chunks(self, filename: str) -> List[Dict]:
        """"Load chunks from a CSV file"""
        import pandas as pd
        
        filepath = f"data/processed/{filename}"
        df = pd.read_csv(filepath, encoding='utf-8')
        
        chunks = df.to_dict('records')
        print(f"✓ Loaded {len(chunks)} chunks from {filepath}")
        
        return chunks