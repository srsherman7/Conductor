import os
import asyncio
import json
from typing import List, Dict, Any, Optional
import anthropic
import openai
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AIModel:
    """Base class for AI model integrations"""
    def __init__(self, name: str):
        self.name = name
    
    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a response from this AI model"""
        raise NotImplementedError("Subclasses must implement this method")

class ClaudeModel(AIModel):
    """Integration with Anthropic's Claude API"""
    def __init__(self, name: str, model_name: str = "claude-3-sonnet-20240229"):
        super().__init__(name)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name
    
    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return {
                "model": self.name,
                "response": response.content[0].text,
                "success": True
            }
        except Exception as e:
            return {
                "model": self.name,
                "error": str(e),
                "success": False
            }

class GPTModel(AIModel):
    """Integration with OpenAI's GPT API"""
    def __init__(self, name: str, model_name: str = "gpt-4"):
        super().__init__(name)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
    
    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return {
                "model": self.name,
                "response": response.choices[0].message.content,
                "success": True
            }
        except Exception as e:
            return {
                "model": self.name,
                "error": str(e),
                "success": False
            }

class GeminiModel(AIModel):
    """Integration with Google's Gemini API"""
    def __init__(self, name: str, model_name: str = "gemini-1.5-pro"):
        super().__init__(name)
        # Configure the Gemini API
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
    
    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        try:
            # Use asyncio to run the synchronous Gemini API in a separate thread
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.generate_content(prompt)
            )
            
            # Extract text from response
            response_text = response.text
            
            return {
                "model": self.name,
                "response": response_text,
                "success": True
            }
        except Exception as e:
            return {
                "model": self.name,
                "error": str(e),
                "success": False
            }

class AIOrchestrator:
    """Orchestrates multiple AI models and compiles their responses"""
    def __init__(self, models: List[AIModel], orchestration_model: Optional[AIModel] = None):
        self.models = models
        # Optional model used to compile the responses
        self.orchestration_model = orchestration_model
    
    async def gather_responses(self, prompt: str) -> List[Dict[str, Any]]:
        """Query all models in parallel and collect their responses"""
        tasks = [model.generate_response(prompt) for model in self.models]
        return await asyncio.gather(*tasks)
    
    async def compile_responses(self, prompt: str, responses: List[Dict[str, Any]]) -> str:
        """Compile multiple responses into a single coherent response"""
        # If we have an orchestration model, use it to compile the responses
        if self.orchestration_model:
            # Create a prompt for the orchestration model
            compilation_prompt = f"""
I have queried multiple AI systems with the following question:
"{prompt}"

Here are their responses:
"""
            for resp in responses:
                if resp["success"]:
                    compilation_prompt += f"\n--- {resp['model']} ---\n{resp['response']}\n"
            
            compilation_prompt += """
Please synthesize these responses into a single coherent response that:
1. Captures the unique insights from each model
2. Resolves any contradictions between the models
3. Provides a comprehensive answer to the original question
4. Cites which model contributed specific information when relevant
"""
            
            result = await self.orchestration_model.generate_response(compilation_prompt)
            if result["success"]:
                return result["response"]
            else:
                return self._fallback_compilation(responses)
        else:
            # Use a simple fallback method if no orchestration model is provided
            return self._fallback_compilation(responses)
    
    def _fallback_compilation(self, responses: List[Dict[str, Any]]) -> str:
        """Simple fallback method to compile responses without using another model"""
        successful_responses = [r for r in responses if r["success"]]
        
        if not successful_responses:
            return "No AI models were able to generate a response."
        
        compiled = "Compiled responses from multiple AI systems:\n\n"
        for resp in successful_responses:
            compiled += f"--- {resp['model']} ---\n{resp['response']}\n\n"
        
        return compiled

async def main():
    # Initialize models
    claude = ClaudeModel("Claude 3 Sonnet")
    gpt = GPTModel("GPT-4")
    gemini = GeminiModel("Gemini 1.5 Pro")
    
    # Use Claude as the orchestration model to compile responses
    claude_opus = ClaudeModel("Claude 3 Opus", "claude-3-opus-20240229")
    
    # Create the orchestrator with all models and the compilation model
    orchestrator = AIOrchestrator([claude, gpt, gemini], claude_opus)
    
    # Example prompt
    prompt = "Explain the concept of neural networks and their applications in simple terms."
    
    # Gather responses from all models
    responses = await orchestrator.gather_responses(prompt)
    
    # Compile the responses into a single response
    final_response = await orchestrator.compile_responses(prompt, responses)
    
    # Print the final, compiled response
    print("\n=== FINAL COMPILED RESPONSE ===\n")
    print(final_response)

if __name__ == "__main__":
    asyncio.run(main())