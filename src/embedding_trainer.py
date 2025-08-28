from sentence_transformers import SentenceTransformer, InputExample, losses

class EmbeddingTrainer:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def prepare_training_data(self, qa_data):
        """آماده‌سازی داده‌های آموزشی"""
        pass
    
    def fine_tune(self, train_examples):
        """Fine-tuning مدل"""
        pass
    
    def save_model(self, path):
        """ذخیره مدل"""
        self.model.save(path)