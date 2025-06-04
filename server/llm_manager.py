from typing import Dict, Optional, Tuple, List, Any, Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from llama_cpp import Llama
import logging
import time

from config import (
    MODEL_ASSIGNMENTS, 
    GLOBAL_LOADING_PARAMETERS,
    GLOBAL_INFERENCE_PARAMETERS,
    validate_parameter,
    get_loading_parameter_defaults,
    get_inference_parameter_defaults
)

logger = logging.getLogger('llm_engine.llm_manager')

@dataclass
class ModelConfig:
    """Model configuration containing inference parameters."""
    name: str
    inference_params: Dict[str, str]
    model_path: str
    default_params: Dict[str, float]
    max_context_window: int = 2048
    loading_params: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.loading_params is None:
            self.loading_params = {}
    
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
        self.loading_parameters = {}  # Store the parameters used to load the model
        
    @classmethod
    def Load(cls, model_name: str, loading_params: Dict[str, Any] = None) -> 'LLMManager':
        """Load a model configuration and initialize the model with custom loading parameters."""
        
        if model_name not in MODEL_ASSIGNMENTS:
            raise ValueError(f"Unknown model: {model_name}")
            
        config = ModelConfig(**MODEL_ASSIGNMENTS[model_name])
        
        # Get default loading parameters
        final_loading_params = get_loading_parameter_defaults()
        
        # Add model-specific loading parameter defaults
        if hasattr(config, 'loading_params') and config.loading_params:
            for param_name, param_def in config.loading_params.items():
                final_loading_params[param_name] = param_def.get("default", final_loading_params.get(param_name))
        
        # Apply user-provided overrides
        if loading_params:
            final_loading_params.update(loading_params)
        
        # Validate all loading parameters
        validated_params = {}
        try:
            # Validate global parameters
            for param_name, value in final_loading_params.items():
                if param_name in GLOBAL_LOADING_PARAMETERS:
                    validated_params[param_name] = validate_parameter(
                        param_name, value, GLOBAL_LOADING_PARAMETERS[param_name]
                    )
                elif hasattr(config, 'loading_params') and param_name in config.loading_params:
                    # Validate model-specific parameters
                    validated_params[param_name] = validate_parameter(
                        param_name, value, config.loading_params[param_name]
                    )
                else:
                    # Unknown parameter - pass through with warning
                    logger.warning(f"Unknown loading parameter: {param_name}")
                    validated_params[param_name] = value
        except ValueError as e:
            raise ValueError(f"Parameter validation failed: {str(e)}")
            
        manager = cls(config)
        manager.loading_parameters = validated_params
        
        # Log the parameters being used
        logger.info(f"Loading model {model_name} with parameters:")
        for param_name, value in validated_params.items():
            logger.info(f"  {param_name}: {value}")
        
        # Load the model with validated parameters
        try:
            manager.llm = Llama(
                model_path=config.model_path,
                n_gpu_layers=validated_params.get("n_gpu_layers", -1),
                n_ctx=validated_params.get("n_ctx", 2048),
                n_threads=validated_params.get("n_threads", 8),
                verbose=False,  # Reduce console output
                use_mlock=validated_params.get("use_mlock", True),
                use_mmap=validated_params.get("use_mmap", True),
                f16_kv=validated_params.get("f16_kv", True)
            )
            logger.info(f"Successfully loaded model: {model_name}")
            return manager
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_loading_parameters(self) -> Dict[str, Any]:
        """Get the loading parameters used for this model instance."""
        return self.loading_parameters.copy()
    
    def get_inference_parameter_defaults(self) -> Dict[str, Any]:
        """Get the default inference parameters for this model."""
        return get_inference_parameter_defaults(self.config.name)
    
    def validate_inference_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and convert inference parameters."""
        validated_params = {}
        
        for param_name, value in params.items():
            if param_name in GLOBAL_INFERENCE_PARAMETERS:
                validated_params[param_name] = validate_parameter(
                    param_name, value, GLOBAL_INFERENCE_PARAMETERS[param_name]
                )
            else:
                # Unknown parameter - pass through with warning
                logger.warning(f"Unknown inference parameter: {param_name}")
                validated_params[param_name] = value
                
        return validated_params
    
    # ============================================================================
    # RAW PROMPT METHODS (UPDATED) - Send prompts exactly as typed by user
    # ============================================================================
    
    def generate_raw(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate a response using the raw prompt without any formatting."""
        if self.llm is None:
            raise ValueError("Model not loaded")
        
        # Combine system and user prompt simply if system prompt is provided
        final_prompt = prompt
        if system_prompt:
            final_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Get default parameters and merge with overrides
        params = self.get_inference_parameter_defaults()
        params.update(kwargs)
        
        # Validate inference parameters
        try:
            validated_params = self.validate_inference_parameters(params)
        except ValueError as e:
            logger.error(f"Invalid inference parameters: {str(e)}")
            raise
        
        # Generate response
        try:
            logger.info(f"Starting raw generation with prompt: {prompt[:50]}...")
            start_time = time.time()
            
            # Send the raw prompt directly to the model
            response = self.llm.create_completion(
                prompt=final_prompt,
                max_tokens=int(validated_params.get("max_tokens", 300)),
                temperature=float(validated_params.get("temperature", 0.7)),
                top_p=float(validated_params.get("top_p", 0.95)),
                top_k=int(validated_params.get("top_k", 40)),
                repeat_penalty=float(validated_params.get("repeat_penalty", 1.1)),
                min_p=float(validated_params.get("min_p", 0.05)),
                stream=False
            )
            
            # Log completion time
            duration = time.time() - start_time
            logger.info(f"Raw generation completed in {duration:.2f} seconds")
            
            # Extract response text
            if isinstance(response, dict) and "choices" in response and len(response["choices"]) > 0:
                raw_response = response["choices"][0]["text"]
            else:
                logger.warning(f"Unexpected response format: {type(response)}")
                if hasattr(response, "get"):
                    raw_response = response.get("text", str(response))
                else:
                    raw_response = str(response)
            
            return raw_response.strip()
            
        except Exception as e:
            logger.error(f"Error generating raw response: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_stream_raw(self, prompt: str, system_prompt: str = "", **kwargs) -> Iterator[Dict[str, Any]]:
        """Generate a streaming response using the raw prompt without any formatting."""
        if self.llm is None:
            raise ValueError("Model not loaded")
        
        # Combine system and user prompt simply if system prompt is provided
        final_prompt = prompt
        if system_prompt:
            final_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Get default parameters and merge with overrides
        params = self.get_inference_parameter_defaults()
        params.update(kwargs)
        
        # Validate inference parameters
        try:
            validated_params = self.validate_inference_parameters(params)
        except ValueError as e:
            logger.error(f"Invalid inference parameters: {str(e)}")
            yield {"error": str(e)}
            return
        
        try:
            logger.info(f"Starting raw streaming generation with prompt: {prompt[:50]}...")
            
            # Create a streaming completion with raw prompt
            stream = self.llm.create_completion(
                prompt=final_prompt,
                max_tokens=int(validated_params.get("max_tokens", 300)),
                temperature=float(validated_params.get("temperature", 0.7)),
                top_p=float(validated_params.get("top_p", 0.95)),
                top_k=int(validated_params.get("top_k", 40)),
                repeat_penalty=float(validated_params.get("repeat_penalty", 1.1)),
                min_p=float(validated_params.get("min_p", 0.05)),
                stream=True
            )
            
            # Return each chunk as it comes in
            collected_text = ""
            for chunk in stream:
                # Extract the text from the chunk
                if isinstance(chunk, dict) and "choices" in chunk and len(chunk["choices"]) > 0:
                    token = chunk["choices"][0].get("text", "")
                else:
                    token = ""
                    if hasattr(chunk, "get"):
                        token = chunk.get("text", "")
                
                collected_text += token
                
                # Yield the token and current collected text
                yield {
                    "token": token,
                    "text": collected_text.strip()
                }
                
        except Exception as e:
            logger.error(f"Error in raw streaming generation: {str(e)}")
            yield {
                "error": str(e)
            }
    
    # ============================================================================
    # FORMATTED PROMPT METHODS (EXISTING) - For future manual formatting features
    # ============================================================================
    
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
        """Generate a response from the model (legacy method for compatibility)."""
        if self.llm is None:
            raise ValueError("Model not loaded")
        
        # Use the provided formatted prompt or create one
        if formatted_prompt is None:
            formatted_prompt = self.format_prompt(prompt, system_prompt)
        
        # Get default parameters and merge with overrides
        params = self.get_inference_parameter_defaults()
        params.update(kwargs)
        
        # Validate inference parameters
        try:
            validated_params = self.validate_inference_parameters(params)
        except ValueError as e:
            logger.error(f"Invalid inference parameters: {str(e)}")
            raise
        
        # Generate response
        try:
            logger.info(f"Starting formatted generation with prompt: {prompt[:50]}...")
            start_time = time.time()
            
            response = self.llm.create_completion(
                prompt=formatted_prompt,
                max_tokens=int(validated_params.get("max_tokens", 300)),
                temperature=float(validated_params.get("temperature", 0.7)),
                top_p=float(validated_params.get("top_p", 0.95)),
                top_k=int(validated_params.get("top_k", 40)),
                repeat_penalty=float(validated_params.get("repeat_penalty", 1.1)),
                min_p=float(validated_params.get("min_p", 0.05)),
                stop=[self.config.assistant_suffix] if self.config.assistant_suffix else None,
                stream=False
            )
            
            # Log completion time
            duration = time.time() - start_time
            logger.info(f"Formatted generation completed in {duration:.2f} seconds")
            
            # Check response structure and extract text
            if isinstance(response, dict) and "choices" in response and len(response["choices"]) > 0:
                raw_response = response["choices"][0]["text"]
            else:
                logger.warning(f"Unexpected response format: {type(response)}")
                if hasattr(response, "get"):
                    raw_response = response.get("text", str(response))
                else:
                    raw_response = str(response)
            
            return self._parse_response(raw_response)
            
        except Exception as e:
            logger.error(f"Error generating formatted response: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_stream(self, prompt: str, system_prompt: str = "", formatted_prompt: str = None, **kwargs) -> Iterator[Dict[str, Any]]:
        """Generate a streaming response from the model (legacy method for compatibility)."""
        if self.llm is None:
            raise ValueError("Model not loaded")
        
        # Use the provided formatted prompt or create one
        if formatted_prompt is None:
            formatted_prompt = self.format_prompt(prompt, system_prompt)
        
        # Get default parameters and merge with overrides
        params = self.get_inference_parameter_defaults()
        params.update(kwargs)
        
        # Validate inference parameters
        try:
            validated_params = self.validate_inference_parameters(params)
        except ValueError as e:
            logger.error(f"Invalid inference parameters: {str(e)}")
            yield {"error": str(e)}
            return
        
        try:
            logger.info(f"Starting formatted streaming generation with prompt: {prompt[:50]}...")
            
            stream = self.llm.create_completion(
                prompt=formatted_prompt,
                max_tokens=int(validated_params.get("max_tokens", 300)),
                temperature=float(validated_params.get("temperature", 0.7)),
                top_p=float(validated_params.get("top_p", 0.95)),
                top_k=int(validated_params.get("top_k", 40)),
                repeat_penalty=float(validated_params.get("repeat_penalty", 1.1)),
                min_p=float(validated_params.get("min_p", 0.05)),
                stop=[self.config.assistant_suffix] if self.config.assistant_suffix else None,
                stream=True
            )
            
            # Return each chunk as it comes in
            collected_text = ""
            for chunk in stream:
                if isinstance(chunk, dict) and "choices" in chunk and len(chunk["choices"]) > 0:
                    token = chunk["choices"][0].get("text", "")
                else:
                    token = ""
                    if hasattr(chunk, "get"):
                        token = chunk.get("text", "")
                
                collected_text += token
                parsed_text = self._parse_response(collected_text)
                
                yield {
                    "token": token,
                    "text": parsed_text
                }
                
        except Exception as e:
            logger.error(f"Error in formatted streaming generation: {str(e)}")
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
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        if self.llm is None:
            # If no model is loaded, provide a rough estimate
            # Most models use roughly 3-4 characters per token on average
            return len(text) // 3
        
        try:
            # Use the model's tokenizer to get accurate count
            tokens = self.llm.tokenize(text.encode('utf-8'))
            return len(tokens)
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}, using rough estimate")
            # Fallback to rough estimate
            return len(text) // 3

    def get_context_usage(self, text: str) -> dict:
        """Get context usage information for any text (raw or formatted)."""
        token_count = self.count_tokens(text)
        
        # Get max context from the model's current settings
        max_context = self.llm.n_ctx() if self.llm else self.loading_parameters.get("n_ctx", 2048)
        
        usage_percentage = (token_count / max_context) * 100 if max_context > 0 else 0
        
        return {
            'token_count': token_count,
            'max_context': max_context,
            'usage_percentage': round(usage_percentage, 1),
            'remaining_tokens': max_context - token_count
        }