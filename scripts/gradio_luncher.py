#!/usr/bin/env python3
"""
Gradio Interface for Persian Drug Information RAG System
Uses the best performing model: distiluse-base-multilingual-cased-v2_finetuned
"""

import gradio as gr
import os
import sys
import time
from typing import List, Tuple, Optional
import json
import warnings

warnings.filterwarnings("ignore")

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import RetrievalSystem
from src.llama_client import LlamaClient
from src.evaluation import RAGEvaluator

class DrugRAGSystem:
    def __init__(self):
        self.retriever = None
        self.llama_client = None
        self.is_initialized = False
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2_finetuned"
        self.chunk_type = "sentence"  # Based on best performance
        self.initialization_status = ""
        
    def initialize_system(self) -> Tuple[bool, str]:
        """Initialize the RAG system with the best performing model"""
        try:
            status_messages = []
            
            # Step 1: Check prerequisites
            status_messages.append("🔍 بررسی فایل‌های مورد نیاز...")
            
            chunk_file = "data/processed/drugs_sentence_chunks.csv"
            faiss_index = f"results/faiss/{self.model_name}_drugs_{self.chunk_type}_chunks.index"
            model_path = f"models/{self.model_name}"
            
            if not os.path.exists(chunk_file):
                return False, f"❌ فایل chunk یافت نشد: {chunk_file}"
            
            if not os.path.exists(faiss_index):
                # Try generic index as fallback
                generic_index = f"results/faiss/drugs_{self.chunk_type}_chunks.index"
                if os.path.exists(generic_index):
                    faiss_index = generic_index
                    status_messages.append(f"⚠️ استفاده از index عمومی: {generic_index}")
                else:
                    return False, f"❌ فایل FAISS index یافت نشد: {faiss_index}"
            
            if not os.path.exists(model_path):
                # Try base model
                base_model = "sentence-transformers/distiluse-base-multilingual-cased-v2"
                model_path = base_model
                status_messages.append(f"⚠️ استفاده از مدل پایه: {base_model}")
            
            status_messages.append("✅ فایل‌های مورد نیاز موجود است")
            
            # Step 2: Initialize LLaMA client
            status_messages.append("🤖 اتصال به سرور LLaMA...")
            
            self.llama_client = LlamaClient("http://127.0.0.1:8080")
            
            # Test LLaMA connection
            test_response = self.llama_client.generate("سلام", max_tokens=5)
            if not test_response:
                return False, "❌ اتصال به سرور LLaMA برقرار نشد. لطفاً سرور را روشن کنید."
            
            status_messages.append("✅ اتصال به LLaMA برقرار شد")
            
            # Step 3: Initialize retrieval system
            status_messages.append("🔍 راه‌اندازی سیستم بازیابی...")
            
            self.retriever = RetrievalSystem(
                method="tfidf",
                model_path=model_path,
                device="cuda" if os.path.exists("/usr/local/cuda") else "cpu"
            )
            
            if not self.retriever.load_chunks_and_index(chunk_file, faiss_index):
                return False, "❌ بارگذاری داده‌ها و index ناموفق بود"
            
            status_messages.append("✅ سیستم بازیابی راه‌اندازی شد")
            
            # Step 4: Final test
            status_messages.append("🧪 تست نهایی سیستم...")
            
            test_contexts, _ = self.retriever.get_contexts_for_rag("آسپرین چیست؟", top_k=3)
            if not test_contexts:
                return False, "❌ تست بازیابی ناموفق بود"
            
            test_answer = self.llama_client.answer_question("آسپرین چیست؟", test_contexts)
            if not test_answer:
                return False, "❌ تست تولید پاسخ ناموفق بود"
            
            status_messages.append("✅ تست نهایی موفق بود")
            status_messages.append("🎉 سیستم آماده پاسخ‌گویی است!")
            
            self.is_initialized = True
            self.initialization_status = "\n".join(status_messages)
            
            return True, self.initialization_status
            
        except Exception as e:
            error_msg = f"❌ خطا در راه‌اندازی: {str(e)}"
            return False, error_msg
    
    def ask_question(self, question: str, num_contexts: int = 5) -> Tuple[str, str, str, List[str]]:
        """Process user question and return answer with details"""
        
        if not self.is_initialized:
            return (
                "⚠️ سیستم هنوز راه‌اندازی نشده است. لطفاً ابتدا سیستم را راه‌اندازی کنید.",
                "",
                "سیستم غیرفعال",
                []
            )
        
        if not question or not question.strip():
            return "لطفاً سوال خود را وارد کنید.", "", "سوال خالی", []
        
        question = question.strip()
        
        try:
            # Step 1: Retrieve relevant contexts
            start_time = time.time()
            contexts, metadata = self.retriever.get_contexts_for_rag(
                question, 
                top_k=num_contexts,
                max_context_length=3000
            )
            retrieval_time = time.time() - start_time
            
            if not contexts:
                return (
                    "متأسفانه اطلاعات مرتبطی در پایگاه داده یافت نشد.",
                    "هیچ متن مرتبطی بازیابی نشد",
                    "بازیابی ناموفق",
                    []
                )
            
            # Step 2: Generate answer
            start_time = time.time()
            answer = self.llama_client.answer_question(question, contexts)
            generation_time = time.time() - start_time
            
            if not answer:
                answer = "متأسفانه قادر به تولید پاسخ نیستم. لطفاً سوال را بازنویسی کنید."
            
            # Step 3: Prepare context information
            context_info = []
            for i, (context, meta) in enumerate(zip(contexts, metadata)):
                context_preview = context[:200] + "..." if len(context) > 200 else context
                context_info.append(f"📄 متن {i+1} (امتیاز: {meta['score']:.4f}):\n{context_preview}")
            
            # Step 4: Prepare processing details
            processing_details = f"""
⏱️ **زمان پردازش:**
- بازیابی متون: {retrieval_time:.3f} ثانیه
- تولید پاسخ: {generation_time:.3f} ثانیه  
- کل: {retrieval_time + generation_time:.3f} ثانیه

🔍 **جزئیات بازیابی:**
- تعداد متون بازیابی شده: {len(contexts)}
- مدل embedding: {self.model_name}
- نوع chunking: {self.chunk_type}

📊 **کیفیت بازیابی:**
- بالاترین امتیاز: {max(m['score'] for m in metadata):.4f}
- پایین‌ترین امتیاز: {min(m['score'] for m in metadata):.4f}
- میانگین امتیاز: {sum(m['score'] for m in metadata) / len(metadata):.4f}
"""
            
            processing_status = f"✅ پردازش موفق در {retrieval_time + generation_time:.3f} ثانیه"
            
            return answer, processing_details, processing_status, context_info
            
        except Exception as e:
            error_msg = f"❌ خطا در پردازش سوال: {str(e)}"
            return error_msg, "", "خطا در پردازش", []

# Initialize RAG system
rag_system = DrugRAGSystem()

def initialize_system_wrapper():
    """Wrapper for system initialization"""
    success, message = rag_system.initialize_system()
    if success:
        return (
            "✅ سیستم با موفقیت راه‌اندازی شد!",
            message,
            gr.update(interactive=True),  # Enable question input
            gr.update(interactive=True),  # Enable ask button
        )
    else:
        return (
            "❌ راه‌اندازی سیستم ناموفق بود",
            message,
            gr.update(interactive=False),  # Keep question input disabled
            gr.update(interactive=False),  # Keep ask button disabled
        )

def ask_question_wrapper(question: str, num_contexts: int):
    """Wrapper for asking questions"""
    answer, details, status, contexts = rag_system.ask_question(question, num_contexts)
    
    # Format contexts for display
    contexts_display = "\n\n".join(contexts) if contexts else "هیچ متن مرتبطی یافت نشد"
    
    return answer, details, status, contexts_display

# Create Gradio interface
with gr.Blocks(
    title="سیستم پاسخ‌گویی هوشمند داروها", 
    theme=gr.themes.Soft(),
    css="""
    .rtl { direction: rtl; text-align: right; }
    .status-success { color: #28a745; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .answer-box { font-size: 16px; line-height: 1.6; }
    """
) as demo:
    
    gr.Markdown("""
    # 🏥 سیستم پاسخ‌گویی هوشمند داروها
    
    این سیستم با استفاده از تکنولوژی RAG (Retrieval-Augmented Generation) پاسخ‌های دقیق و علمی در مورد داروها ارائه می‌دهد.
    
    **مشخصات فنی:**
    - مدل embedding: paraphrase-multilingual-MiniLM-L12-v2_finetuned
    - مدل زبان: LLaMA 3.2 1B finetuned (local with llama.cpp)
    - پایگاه داده: Drugs.pdf
    """, elem_classes=["rtl"])
    
    with gr.Row():
        with gr.Column(scale=2):
            # Initialization section
            gr.Markdown("## 🚀 راه‌اندازی سیستم", elem_classes=["rtl"])
            
            with gr.Row():
                init_button = gr.Button("🔧 راه‌اندازی سیستم", variant="primary", size="lg")
            
            init_status = gr.Textbox(
                label="وضعیت راه‌اندازی",
                value="سیستم هنوز راه‌اندازی نشده است",
                elem_classes=["rtl"],
                interactive=False
            )
            
            init_details = gr.Textbox(
                label="جزئیات راه‌اندازی", 
                lines=8,
                elem_classes=["rtl"],
                interactive=False
            )
            
            # Question section
            gr.Markdown("## 💬 سوال و جواب", elem_classes=["rtl"])
            
            question_input = gr.Textbox(
                label="سوال شما",
                placeholder="مثال: آسپرین چه عوارضی دارد؟",
                lines=2,
                elem_classes=["rtl"],
                interactive=False  # Disabled until initialization
            )
            
            with gr.Row():
                num_contexts = gr.Slider(
                    minimum=3,
                    maximum=10,
                    value=5,
                    step=1,
                    label="تعداد متون مرجع",
                    elem_classes=["rtl"]
                )
                
                ask_button = gr.Button(
                    "🔍 پرسش",
                    variant="secondary",
                    interactive=False  # Disabled until initialization
                )
        
        with gr.Column(scale=3):
            # Answer section
            gr.Markdown("## 🤖 پاسخ سیستم", elem_classes=["rtl"])
            
            answer_output = gr.Textbox(
                label="پاسخ",
                lines=8,
                elem_classes=["rtl", "answer-box"],
                interactive=False,
                rtl=True  # اضافه کردن این خط
            )

            processing_status = gr.Textbox(
                label="وضعیت پردازش",
                elem_classes=["rtl"],
                interactive=False
            )
            
            # Processing details (collapsible)
            with gr.Accordion("📊 جزئیات پردازش", open=False):
                processing_details = gr.Textbox(
                    label="جزئیات فنی",
                    lines=10,
                    elem_classes=["rtl"],
                    interactive=False
                )
            
            # Retrieved contexts (collapsible)
            with gr.Accordion("📄 متون مرجع", open=False):
                contexts_output = gr.Textbox(
                    label="متون بازیابی شده",
                    lines=15,
                    elem_classes=["rtl"],
                    interactive=False,
                    rtl=True # اضافه کردن این خط
                )
    
    # Event handlers
    init_button.click(
        initialize_system_wrapper,
        outputs=[init_status, init_details, question_input, ask_button]
    )
    
    ask_button.click(
        ask_question_wrapper,
        inputs=[question_input, num_contexts],
        outputs=[answer_output, processing_details, processing_status, contexts_output]
    )
    
    # Enable enter key for asking questions
    question_input.submit(
        ask_question_wrapper,
        inputs=[question_input, num_contexts], 
        outputs=[answer_output, processing_details, processing_status, contexts_output]
    )
    

# Launch configuration
if __name__ == "__main__":
    print("🚀 Starting Drug RAG System Interface...")
    print("🌐 Interface will be available at: http://localhost:7860")
    print("🔧 Make sure LLaMA server is running on localhost:8080")
    print("📁 Make sure all required files are in place")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True if you want public sharing
        show_error=True,
        # show_tips=True,
        # enable_queue=True,
        max_threads=10
    )