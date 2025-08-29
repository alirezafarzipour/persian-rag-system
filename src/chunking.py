# chunking.py
from .utils import PersianTextProcessor
from typing import List, Dict, Tuple, Generator
import math
import gc
import os

class TextChunker:
    def __init__(self, config):
        self.config = config
        self.text_processor = PersianTextProcessor()
        
    def word_based_chunking_generator(self, text: str) -> Generator[Dict[str, any], None, None]:
        """Word-based chunking با generator برای صرفه‌جویی در حافظه"""
        print("Performing memory-optimized word-based chunking...")
        
        chunk_size = self.config['chunking']['word_chunk_size']
        overlap = self.config['chunking']['word_overlap']
        
        # پردازش متن به قسمت‌های کوچک
        text_length = len(text)
        max_segment_size = 50000  # 50K کاراکتر در هر segment
        
        chunk_id = 0
        total_chunks = 0
        
        for segment_start in range(0, text_length, max_segment_size - 5000):  # overlap بین segments
            segment_end = min(segment_start + max_segment_size, text_length)
            segment_text = text[segment_start:segment_end]
            
            # نرمال‌سازی segment
            normalized_segment = self.text_processor.normalize_text(segment_text)
            
            # tokenization به صورت lazy
            words = self._tokenize_words_lazy(normalized_segment)
            
            # تولید chunks از این segment
            start_idx = 0
            segment_word_count = 0
            
            current_chunk_words = []
            
            for word in words:
                current_chunk_words.append(word)
                segment_word_count += 1
                
                # اگر chunk پر شد
                if len(current_chunk_words) >= chunk_size:
                    chunk_text = ' '.join(current_chunk_words)
                    
                    chunk_info = {
                        'id': f'word_chunk_{chunk_id}',
                        'text': chunk_text,
                        'start_word': start_idx,
                        'end_word': start_idx + len(current_chunk_words),
                        'num_words': len(current_chunk_words),
                        'chunk_type': 'word_based',
                        'overlap_words': overlap if chunk_id > 0 else 0
                    }
                    
                    yield chunk_info
                    total_chunks += 1
                    chunk_id += 1
                    
                    # حفظ overlap words
                    if overlap > 0:
                        current_chunk_words = current_chunk_words[-overlap:]
                        start_idx += (chunk_size - overlap)
                    else:
                        current_chunk_words = []
                        start_idx += chunk_size
            
            # اگر words باقی مانده داریم
            if current_chunk_words and len(current_chunk_words) >= 10:  # حداقل 10 کلمه
                chunk_text = ' '.join(current_chunk_words)
                
                chunk_info = {
                    'id': f'word_chunk_{chunk_id}',
                    'text': chunk_text,
                    'start_word': start_idx,
                    'end_word': start_idx + len(current_chunk_words),
                    'num_words': len(current_chunk_words),
                    'chunk_type': 'word_based',
                    'overlap_words': 0
                }
                
                yield chunk_info
                total_chunks += 1
                chunk_id += 1
            
            # آزاد کردن حافظه segment
            del normalized_segment, words, current_chunk_words
            gc.collect()
            
            print(f"  Processed segment {segment_start//max_segment_size + 1}, chunks so far: {total_chunks}")
    
    def _tokenize_words_lazy(self, text: str) -> Generator[str, None, None]:
        """Lazy word tokenization برای صرفه‌جویی در حافظه"""
        try:
            # استفاده از hazm tokenizer
            tokens = self.text_processor.tokenize_words(text)
            for token in tokens:
                yield token
        except Exception as e:
            print(f"Warning: hazm tokenization failed ({e}), using simple split")
            # fallback به split ساده
            for word in text.split():
                yield word.strip()
    
    def word_based_chunking(self, text: str) -> List[Dict[str, any]]:
        """Word-based chunking با مدیریت حافظه"""
        chunks = []
        
        try:
            # استفاده از generator
            for chunk in self.word_based_chunking_generator(text):
                chunks.append(chunk)
                
                # ذخیره chunks به صورت دوره‌ای برای جلوگیری از مصرف بالای حافظه
                if len(chunks) % 1000 == 0:
                    print(f"  Generated {len(chunks)} chunks so far...")
                    gc.collect()
                    
        except Exception as e:
            print(f"Error in word chunking: {e}")
            return []
        
        print(f"✓ Created {len(chunks)} word-based chunks")
        return chunks
    
    def sentence_based_chunking(self, text: str) -> List[Dict[str, any]]:
        """Sentence-based chunking با بهینه‌سازی حافظه"""
        print("Performing memory-optimized sentence-based chunking...")
        
        sentences_per_chunk = self.config['chunking']['sentences_per_chunk']
        
        chunks = []
        chunk_id = 0
        
        # پردازش متن به قسمت‌های کوچک
        text_length = len(text)
        max_segment_size = 100000  # 100K کاراکتر در هر segment
        
        for segment_start in range(0, text_length, max_segment_size - 10000):  # overlap
            segment_end = min(segment_start + max_segment_size, text_length)
            segment_text = text[segment_start:segment_end]
            
            # نرمال‌سازی segment
            normalized_segment = self.text_processor.normalize_text(segment_text)
            
            # جملات از این segment
            try:
                sentences = self.text_processor.tokenize_sentences(normalized_segment)
            except Exception as e:
                print(f"Warning: sentence tokenization failed ({e}), using simple split")
                # fallback به تقسیم با نقطه
                sentences = [s.strip() + '.' for s in normalized_segment.split('.') if s.strip()]
            
            # تولید chunks از sentences
            for i in range(0, len(sentences), sentences_per_chunk):
                chunk_sentences = sentences[i:i + sentences_per_chunk]
                
                if not chunk_sentences:
                    continue
                    
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
                
                # آزاد کردن حافظه هر 500 chunk
                if len(chunks) % 500 == 0:
                    print(f"  Generated {len(chunks)} sentence chunks so far...")
                    gc.collect()
            
            # آزاد کردن حافظه segment
            del normalized_segment, sentences
            gc.collect()
            
            print(f"  Processed segment {segment_start//max_segment_size + 1}")
        
        print(f"✓ Created {len(chunks)} sentence-based chunks")
        return chunks
    
    def process_pdf_document(self, pdf_text: str) -> Tuple[List[Dict], List[Dict]]:
        """Full PDF processing با مدیریت حافظه"""
        print("Processing PDF document with memory-optimized chunking...")
        
        if not pdf_text or len(pdf_text.strip()) < 100:
            print("Warning: PDF text is too short or empty")
            return [], []
        
        print(f"PDF text length: {len(pdf_text)} characters")
        
        # Word-based chunking
        print("\n--- Word-based chunking ---")
        word_chunks = self.word_based_chunking(pdf_text)
        
        # آزاد کردن حافظه قبل از sentence chunking
        gc.collect()
        
        # Sentence-based chunking  
        print("\n--- Sentence-based chunking ---")
        sentence_chunks = self.sentence_based_chunking(pdf_text)
        
        # آمار نهایی
        total_words = len(pdf_text.split())
        
        print(f"\n✓ PDF processing completed:")
        print(f"  - Total characters: {len(pdf_text)}")
        print(f"  - Estimated total words: {total_words}")
        print(f"  - Word-based chunks: {len(word_chunks)}")
        print(f"  - Sentence-based chunks: {len(sentence_chunks)}")
        
        return word_chunks, sentence_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict[str, any]:
        """محاسبه آمار chunks"""
        if not chunks:
            return {}
        
        # محاسبه آمار به صورت batch برای صرفه‌جویی در حافظه
        total_words = 0
        word_counts = []
        
        batch_size = 1000
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_word_counts = []
            
            for chunk in batch_chunks:
                word_count = len(chunk['text'].split())
                word_counts.append(word_count)
                total_words += word_count
                batch_word_counts.append(word_count)
            
            # آزاد کردن حافظه batch
            del batch_chunks, batch_word_counts
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        stats = {
            'total_chunks': len(chunks),
            'avg_words_per_chunk': total_words / len(chunks),
            'min_words_per_chunk': min(word_counts),
            'max_words_per_chunk': max(word_counts),
            'total_words': total_words,
            'chunk_type': chunks[0]['chunk_type'] if chunks else 'unknown'
        }
        
        # آزاد کردن حافظه
        del word_counts
        gc.collect()
        
        return stats
    
    def save_chunks(self, chunks: List[Dict], filename: str):
        """ذخیره chunks به صورت بهینه"""
        import pandas as pd
        
        os.makedirs('data/processed', exist_ok=True)
        filepath = f"data/processed/{filename}"
        
        # ذخیره به صورت batch برای جلوگیری از مصرف بالای حافظه
        batch_size = 5000
        
        if len(chunks) <= batch_size:
            # اگر chunks کم باشد، مستقیم ذخیره کن
            df = pd.DataFrame(chunks)
            df.to_csv(filepath, index=False, encoding='utf-8')
        else:
            # ذخیره به صورت batch
            print(f"  Saving {len(chunks)} chunks in batches...")
            
            first_batch = True
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                df_batch = pd.DataFrame(batch_chunks)
                
                if first_batch:
                    df_batch.to_csv(filepath, index=False, encoding='utf-8', mode='w')
                    first_batch = False
                else:
                    df_batch.to_csv(filepath, index=False, encoding='utf-8', mode='a', header=False)
                
                # آزاد کردن حافظه
                del batch_chunks, df_batch
                gc.collect()
                
                if i % (batch_size * 2) == 0:
                    print(f"    Saved {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
        
        print(f"✓ Chunks saved to {filepath}")
    
    def load_chunks(self, filename: str) -> List[Dict]:
        """بارگذاری chunks به صورت بهینه"""
        import pandas as pd
        
        filepath = f"data/processed/{filename}"
        
        try:
            # بارگذاری به صورت batch
            chunk_size = 5000
            chunks = []
            
            # خواندن فایل به قسمت‌ها
            df_iterator = pd.read_csv(filepath, encoding='utf-8', chunksize=chunk_size)
            
            batch_num = 1
            for df_batch in df_iterator:
                batch_chunks = df_batch.to_dict('records')
                chunks.extend(batch_chunks)
                
                print(f"  Loaded batch {batch_num}, total chunks: {len(chunks)}")
                batch_num += 1
                
                # آزاد کردن حافظه
                del df_batch, batch_chunks
                gc.collect()
            
            print(f"✓ Loaded {len(chunks)} chunks from {filepath}")
            return chunks
            
        except Exception as e:
            print(f"Error loading chunks: {e}")
            return []