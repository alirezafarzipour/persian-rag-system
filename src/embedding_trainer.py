from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import random
from typing import List, Dict
import os
import torch
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

class EmbeddingTrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"Loaded model: {model_name}")
    
    def prepare_training_data(self, qa_data: List[Dict]) -> List[InputExample]:
        print("Preparing training examples...")
        
        training_examples = []
        
        for item in qa_data:
            question = item['question']
            answer = item['answer']
            context = item.get('context', '')
            
            if not question or not answer:
                continue
            
            training_examples.append(
                InputExample(texts=[question, answer], label=1.0)
            )
            
            if context and len(context.strip()) > 10:
                training_examples.append(
                    InputExample(texts=[question, context], label=0.8)
                )
        
        negative_examples = self._create_negative_examples(qa_data)
        training_examples.extend(negative_examples)
        
        random.shuffle(training_examples)
        print(f"Created {len(training_examples)} training examples")
        
        return training_examples
    
    def _create_negative_examples(self, qa_data: List[Dict], num_negatives: int = None) -> List[InputExample]:
        if num_negatives is None:
            num_negatives = min(len(qa_data) // 2, 1000)
        
        negative_examples = []
        questions = [item['question'] for item in qa_data if item['question']]
        answers = [item['answer'] for item in qa_data if item['answer']]
        
        for _ in range(num_negatives):
            question = random.choice(questions)
            wrong_answer = random.choice(answers)
            
            attempts = 0
            while wrong_answer in [item['answer'] for item in qa_data if item['question'] == question] and attempts < 10:
                wrong_answer = random.choice(answers)
                attempts += 1
            
            negative_examples.append(
                InputExample(texts=[question, wrong_answer], label=0.0)
            )
        
        return negative_examples
    
    def prepare_evaluation_data(self, test_data: List[Dict]) -> List[InputExample]:
        eval_examples = []
        
        # positive examples
        for item in test_data[:100]:  # limit for speed
            question = item['question']
            answer = item['answer']
            
            if question and answer:
                eval_examples.append(
                    InputExample(texts=[question, answer], label=1.0)
                )
        
        questions = [item['question'] for item in test_data[:50] if item['question']]
        answers = [item['answer'] for item in test_data[:50] if item['answer']]
        
        for i in range(min(50, len(questions))):  
            question = questions[i]
            wrong_answer = answers[(i + len(answers)//2) % len(answers)]
            
            eval_examples.append(
                InputExample(texts=[question, wrong_answer], label=0.0)
            )
        
        print(f"Created {len(eval_examples)} evaluation examples")
        return eval_examples
    
    def fine_tune(self, train_examples: List[InputExample], 
                  eval_examples: List[InputExample] = None,
                  epochs: int = 3, batch_size: int = 16, 
                  warmup_steps: int = 100) -> str:
        """Fine-tuning مدل embedding"""
        
        print(f"Fine-tuning {self.model_name}...")
        print(f"Training examples: {len(train_examples)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        model_save_path = f"models/{self.model_name.split('/')[-1]}_finetuned"
        os.makedirs("models", exist_ok=True)
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            save_best_model=True,
            show_progress_bar=False  
        )
        
        print(f"✓ Fine-tuning completed! Model saved to {model_save_path}")
        return model_save_path
    
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        self.model = SentenceTransformer(path)
        print(f"Model loaded from {path}")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    def get_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.model.encode([text1, text2])
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(embeddings[0]).unsqueeze(0),
            torch.tensor(embeddings[1]).unsqueeze(0)
        )
        return similarity.item()