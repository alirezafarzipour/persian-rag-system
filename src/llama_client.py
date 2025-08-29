import requests
import json
import time
from typing import List, Dict, Optional

class LlamaClient:
    """Cliemt for LLaMA 3.2 local server (llama.cpp)"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        if self._test_connection():
            print(f"✓ Connected to LLaMA server at {self.base_url}")
        else:
            print(f"⚠ Could not connect to LLaMA server at {self.base_url}")
    
    def _test_connection(self) -> bool:
        """Server connection test"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            try:
                response = self.session.get(f"{self.base_url}/v1/models", timeout=5)
                return response.status_code in [200, 404] 
            except:
                return False
    
    def generate(self, prompt: str, max_tokens: int = 512, 
                temperature: float = 0.1, top_p: float = 0.9,
                stop: Optional[List[str]] = None) -> Optional[str]:
        """Gen answer with LLaMA"""
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "stop": stop or ["</s>", "<|eot_id|>", "\n\nسوال:", "\n\nپرسش:"]
        }
        
        try:
            response = self._try_completion_endpoint(payload)
            if response:
                return response
                
            response = self._try_chat_endpoint(prompt, payload)
            if response:
                return response
            
            print("Could not get response from any endpoint")
            return None
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
    
    def _try_completion_endpoint(self, payload: Dict) -> Optional[str]:
        """Try completion endpoint"""
        try:
            response = self.session.post(
                f"{self.base_url}/completion",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "content" in data:
                    return data["content"].strip()
                elif "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["text"].strip()
                    
        except Exception as e:
            print(f"Completion endpoint failed: {e}")
        
        return None
    
    def _try_chat_endpoint(self, prompt: str, payload: Dict) -> Optional[str]:
        """Try chat endpoint"""
        try:
            chat_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": payload.get("max_tokens", 512),
                "temperature": payload.get("temperature", 0.1),
                "top_p": payload.get("top_p", 0.9),
                "stream": False
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=chat_payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
            
            response = self.session.post(
                f"{self.base_url}/chat",
                json=chat_payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if "content" in data:
                    return data["content"].strip()
                elif "response" in data:
                    return data["response"].strip()
                    
        except Exception as e:
            print(f"Chat endpoint failed: {e}")
        
        return None
    
    def create_rag_prompt(self, question: str, contexts: List[str], 
                         max_context_length: int = 2000) -> str:
        """Create prompt for RAG"""
        
        # ترکیب contexts
        combined_context = ""
        current_length = 0
        
        for i, context in enumerate(contexts):
            context_text = f"متن {i+1}: {context}\n\n"
            if current_length + len(context_text) > max_context_length:
                break
            combined_context += context_text
            current_length += len(context_text)
        
        # template فارسی برای RAG
        prompt = f"""بر اساس اطلاعات ارائه شده، به سوال پاسخ دهید.

اطلاعات مرجع:
{combined_context.strip()}

سوال: {question}

پاسخ: بر اساس اطلاعات ارائه شده، """
        
        return prompt
    
    def answer_question(self, question: str, contexts: List[str], 
                       max_tokens: int = 256, temperature: float = 0.1) -> Optional[str]:
        """Answering the question using RAG"""
        
        prompt = self.create_rag_prompt(question, contexts)
        
        response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "<|eot_id|>", "\n\nسوال:", "\n\nپرسش:", "\n\nQuestion:"]
        )
        
        if response:
            # پاک‌سازی پاسخ
            response = response.strip()
            
            # remove repeat prompt in answer
            if "پاسخ:" in response:
                response = response.split("پاسخ:")[-1].strip()
            
            # need to be more...
            prefixes_to_remove = [
                "بر اساس اطلاعات ارائه شده، ",
                "بر اساس متن، ",
                "طبق اطلاعات، ",
                "با توجه به متن، "
            ]
            
            for prefix in prefixes_to_remove:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            
            return response
        
        return None
    
    def batch_answer(self, questions_contexts: List[Dict], 
                    max_tokens: int = 256, temperature: float = 0.1,
                    delay_between_requests: float = 0.5) -> List[Optional[str]]:
        """Batch answer"""
        
        answers = []
        
        for i, item in enumerate(questions_contexts):
            print(f"  Processing question {i+1}/{len(questions_contexts)}")
            
            question = item['question']
            contexts = item['contexts']
            
            answer = self.answer_question(
                question=question,
                contexts=contexts,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answers.append(answer)
            
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)
        
        return answers
    
    def get_server_info(self) -> Dict:
        """دریافت اطلاعات server"""
        info = {
            "status": "unknown",
            "base_url": self.base_url,
            "endpoints": []
        }
        
        # تست endpoints مختلف
        endpoints_to_test = [
            "/health",
            "/v1/models", 
            "/completion",
            "/chat",
            "/v1/chat/completions"
        ]
        
        for endpoint in endpoints_to_test:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code in [200, 405]:  # 405 = Method not allowed but endpoint exists
                    info["endpoints"].append(endpoint)
            except:
                pass
        
        if info["endpoints"]:
            info["status"] = "connected"
        else:
            info["status"] = "disconnected"
            
        return info
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'session'):
            self.session.close()