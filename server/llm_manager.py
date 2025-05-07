from typing import Dict, Optional, Tuple, List, Any, Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from llama_cpp import Llama
import logging
import time

from config import MODEL_ASSIGNMENTS, DEFAULT_N_GPU_LAYERS, DEFAULT_N_CTX, DEFAULT_N_THREADS

logger = logging.getLogger('llm_engine.llm_manager')

@dataclass
class ModelConfig:
    """Model configuration containing inference parameters."""
    name: str
    inference_params: Dict[str, str]
    model_path: str
    default_params: Dict[str, float]
    max_context_window: int = DEFAULT_N_CTX
    
    @property
    def system_prefix(self) -> str:
        return self.inference_params.get('pre_prompt_prefix', '')
        
    @property
    def system_suffix(self) -> str:
        return self.inference_params.get('pre_prompt_suffix', '')
        
    @property
    def user_prefix(self) -> str:
        return self.inference_params.get('input_prefix', '')
        
    @property
    def user_suffix(self) -> str:
        return self.inference_params.get('input_suffix', '')
        
    @property
    def assistant_prefix(self) -> str:
        return self.inference_params.get('assistant_prefix', '')
        
    @property
    def assistant_suffix(self) -> str:
        return self.inference_params.get('assistant_suffix', '')

class LLMManager:
    """Manages LLM interactions, prompt formatting, and model operations."""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.llm = None
        
    @classmethod
    def Load(cls, model_name: str, context_length: int = None) -> 'LLMManager':
        """Load a model configuration and initialize the model."""
        
        if model_name not in MODEL_ASSIGNMENTS:
            raise ValueError(f"Unknown model: {model_name}")
            
        config = ModelConfig(**MODEL_ASSIGNMENTS[model_name])
        
        # If a context length was specified, override the default
        if context_length is not None:
            # Ensure the context length doesn't exceed the model's max context window
            if hasattr(config, 'max_context_window') and context_length > config.max_context_window:
                logger.warning(
                    f"Requested context length {context_length} exceeds model's max context window "
                    f"of {config.max_context_window}. Using {config.max_context_window} instead."
                )
                context_length = config.max_context_window
            logger.info(f"Setting custom context length: {context_length}")
        else:
            # Use the default context length from config
            context_length = DEFAULT_N_CTX
            logger.info(f"Using default context length: {context_length}")
            
        manager = cls(config)
        
        # Load the model
        try:
            manager.llm = Llama(
                model_path=config.model_path,
                n_gpu_layers=DEFAULT_N_GPU_LAYERS,
                n_ctx=context_length,
                n_threads=DEFAULT_N_THREADS,
                verbose=False,  # Reduce console output
                use_mlock=True,  # Keep the model in memory
                use_mmap=True,   # Use memory mapping
                f16_kv=True      # Use half-precision for KV cache
            )
            logger.info(f"Successfully loaded model: {model_name} with context length {context_length}")
            return manager
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
    def format_prompt(self, prompt: str, system_prompt: str = "") -> str:
        """Format a prompt with system and user content."""
        messages = {}
        if system_prompt:
            messages["system"] = system_prompt
        messages["user"] = prompt
        return self.wrap_multiple(messages)
    
    def wrap(self, role: str, content: str) -> str:
        """Wrap content with appropriate prefixes/suffixes based on role."""
        if role == "system":
            return f"{self.config.system_prefix}{content}{self.config.system_suffix}"
        elif role == "user":
            return f"{self.config.user_prefix}{content}{self.config.user_suffix}"
        elif role == "assistant":
            return f"{self.config.assistant_prefix}{content}{self.config.assistant_suffix}"
        else:
            raise ValueError(f"Unknown role: {role}")
            
    def wrap_multiple(self, messages: Dict[str, str]) -> str:
        """Wrap multiple messages in sequence."""
        result = []
        for role, content in messages.items():
            result.append(self.wrap(role, content))
        return "".join(result)
    
    def generate(self, prompt: str, system_prompt: str = "", formatted_prompt: str = None, **kwargs) -> str:
        """Generate a response from the model."""
        if self.llm is None:
            raise ValueError("Model not loaded")
        
        # Use the provided formatted prompt or create one
        if formatted_prompt is None:
            formatted_prompt = self.format_prompt(prompt, system_prompt)
        
        # Merge default parameters with any provided overrides
        params = {**self.config.default_params}
        params.update(kwargs)
        
        # Generate response
        try:
            # Log that we're starting generation
            logger.info(f"Starting generation with prompt: {prompt[:50]}...")
            start_time = time.time()
            
            # Use the completion method to get a single response
            response = self.llm.create_completion(
                prompt=formatted_prompt,
                max_tokens=int(params.get("max_tokens", 300)),
                temperature=float(params.get("temperature", 0.7)),
                top_p=float(params.get("top_p", 0.95)),
                top_k=int(params.get("top_k", 40)),
                repeat_penalty=float(params.get("repeat_penalty", 1.1)),
                stop=[self.config.assistant_suffix] if self.config.assistant_suffix else None,
                stream=False
            )
            
            # Log completion time
            duration = time.time() - start_time
            logger.info(f"Generation completed in {duration:.2f} seconds")
            
            # Check response structure and extract text
            if isinstance(response, dict) and "choices" in response and len(response["choices"]) > 0:
                raw_response = response["choices"][0]["text"]
            else:
                logger.warning(f"Unexpected response format: {type(response)}")
                if hasattr(response, "get"):  # Check if it's dict-like
                    raw_response = response.get("text", str(response))
                else:
                    raw_response = str(response)
            
            return self._parse_response(raw_response)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_stream(self, prompt: str, system_prompt: str = "", formatted_prompt: str = None, **kwargs) -> Iterator[Dict[str, Any]]:
        """Generate a streaming response from the model."""
        if self.llm is None:
            raise ValueError("Model not loaded")
        
        # Use the provided formatted prompt or create one
        if formatted_prompt is None:
            formatted_prompt = self.format_prompt(prompt, system_prompt)
        
        # Merge default parameters with any provided overrides
        params = {**self.config.default_params}
        params.update(kwargs)
        
        try:
            # Log that we're starting generation
            logger.info(f"Starting streaming generation with prompt: {prompt[:50]}...")
            
            # Create a streaming completion
            stream = self.llm.create_completion(
                prompt=formatted_prompt,
                max_tokens=int(params.get("max_tokens", 300)),
                temperature=float(params.get("temperature", 0.7)),
                top_p=float(params.get("top_p", 0.95)),
                top_k=int(params.get("top_k", 40)),
                repeat_penalty=float(params.get("repeat_penalty", 1.1)),
                stop=[self.config.assistant_suffix] if self.config.assistant_suffix else None,
                stream=True
            )
            
            # Return each chunk as it comes in
            collected_text = ""
            for chunk in stream:
                # Extract the text from the chunk
                if isinstance(chunk, dict) and "choices" in chunk and len(chunk["choices"]) > 0:
                    token = chunk["choices"][0].get("text", "")
                else:
                    # Handle other chunk formats if needed
                    token = ""
                    if hasattr(chunk, "get"):
                        token = chunk.get("text", "")
                
                collected_text += token
                parsed_text = self._parse_response(collected_text)
                
                # Yield the token and current collected text
                yield {
                    "token": token,
                    "text": parsed_text
                }
                
        except Exception as e:
            logger.error(f"Error in streaming generation: {str(e)}")
            yield {
                "error": str(e)
            }
    
    def _parse_response(self, response: str) -> str:
        """Parse the response to extract just the assistant's message."""
        # Some models might include the assistant prefix in the response
        # We want to remove that if present
        if self.config.assistant_prefix and response.startswith(self.config.assistant_prefix):
            response = response[len(self.config.assistant_prefix):]
            
        # Remove assistant suffix if present at the end
        if self.config.assistant_suffix and response.endswith(self.config.assistant_suffix):
            response = response[:-len(self.config.assistant_suffix)]
            
        return response.strip()