from typing import Dict, Any, Iterator
from dataclasses import dataclass
from llama_cpp import Llama
import logging
import threading
import time

from config import (
    MODEL_ASSIGNMENTS,
    GLOBAL_LOADING_PARAMETERS,
    GLOBAL_INFERENCE_PARAMETERS,
    LOG_PROMPT_CONTENT,
    ParameterValidationError,
    validate_parameter,
    get_loading_parameter_defaults,
    get_inference_parameter_defaults,
    get_default_n_ctx
)

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration containing inference parameters."""
    name: str
    inference_params: Dict[str, str]
    model_path: str
    default_params: Dict[str, float]
    max_context_window: int = 2048
    loading_params: Dict[str, Dict] = None
    # The MODEL_ASSIGNMENTS key this config came from ('name' is only a display name)
    key: str = ""

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

def build_model_config(model_name: str) -> ModelConfig:
    """Build a ModelConfig for a MODEL_ASSIGNMENTS key."""
    if model_name not in MODEL_ASSIGNMENTS:
        raise ValueError(f"Unknown model: {model_name}")
    config = ModelConfig(**MODEL_ASSIGNMENTS[model_name])
    config.key = model_name
    return config

class LLMManager:
    """Manages LLM interactions, prompt formatting, and model operations."""

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.llm = None
        self.loading_parameters = {}  # Store the parameters used to load the model
        # Guards the *lifetime* of self.llm only - held by close() around the
        # native free, and by the lock-free readers (count_tokens,
        # get_max_context) across their native calls. Deliberately NOT the
        # server's model_lock: token counting must stay concurrent with
        # generation, it just must never overlap a llama_free_model.
        self._instance_lock = threading.Lock()

    @classmethod
    def Load(cls, model_name: str, loading_params: Dict[str, Any] = None) -> 'LLMManager':
        """Load a model configuration and initialize the model with custom loading parameters."""

        config = build_model_config(model_name)

        # Get default loading parameters
        final_loading_params = get_loading_parameter_defaults()

        # Add model-specific loading parameter defaults
        for param_name, param_def in config.loading_params.items():
            final_loading_params[param_name] = param_def.get("default", final_loading_params.get(param_name))

        # Apply user-provided overrides
        if loading_params:
            final_loading_params.update(loading_params)

        # Validate all loading parameters
        validated_params = {}
        try:
            for param_name, value in final_loading_params.items():
                if param_name in GLOBAL_LOADING_PARAMETERS:
                    validated_params[param_name] = validate_parameter(
                        param_name, value, GLOBAL_LOADING_PARAMETERS[param_name]
                    )
                elif param_name in config.loading_params:
                    # Validate model-specific parameters
                    validated_params[param_name] = validate_parameter(
                        param_name, value, config.loading_params[param_name]
                    )
                else:
                    # Unknown parameter - pass through with warning
                    logger.warning(f"Unknown loading parameter: {param_name}")
                    validated_params[param_name] = value
        except ValueError as e:
            raise ParameterValidationError(f"Parameter validation failed: {str(e)}")

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
                n_ctx=validated_params.get("n_ctx", get_default_n_ctx(model_name)),
                n_threads=validated_params.get("n_threads", 8),
                verbose=False,  # Reduce console output
                use_mlock=validated_params.get("use_mlock", True),
                use_mmap=validated_params.get("use_mmap", True)
            )
            logger.info(f"Successfully loaded model: {model_name}")
            return manager
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            manager.close()
            raise

    def close(self) -> None:
        """Release the model's resources (VRAM included) and drop the reference."""
        # Held across the free so a reader can't be inside a native call on this
        # instance while llama_free_model runs underneath it.
        with self._instance_lock:
            llm = self.llm
            self.llm = None
            if llm is None:
                return
            try:
                llm.close()
            except Exception as e:
                logger.warning(f"Error closing model {self.config.key}: {str(e)}")

    def get_loading_parameters(self) -> Dict[str, Any]:
        """Get the loading parameters used for this model instance."""
        return self.loading_parameters.copy()

    def get_inference_parameter_defaults(self) -> Dict[str, Any]:
        """Get the default inference parameters for this model."""
        return get_inference_parameter_defaults(self.config.key)

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
    # RAW PROMPT METHODS - Send prompts exactly as typed by user
    # ============================================================================

    def _completion_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge defaults with per-request overrides and map them onto create_completion."""
        params = self.get_inference_parameter_defaults()
        params.update(kwargs)
        validated_params = self.validate_inference_parameters(params)

        return {
            "max_tokens": int(validated_params.get("max_tokens", 300)),
            "temperature": float(validated_params.get("temperature", 0.7)),
            "top_p": float(validated_params.get("top_p", 0.95)),
            "top_k": int(validated_params.get("top_k", 40)),
            "repeat_penalty": float(validated_params.get("repeat_penalty", 1.1)),
            "min_p": float(validated_params.get("min_p", 0.05)),
        }

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        """Pull the text out of a create_completion chunk."""
        if isinstance(chunk, dict) and "choices" in chunk and len(chunk["choices"]) > 0:
            return chunk["choices"][0].get("text", "")
        if hasattr(chunk, "get"):
            return chunk.get("text", "")
        return ""

    def generate_raw(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate a response using the raw prompt without any formatting."""
        if self.llm is None:
            raise ValueError("Model not loaded")

        # Combine system and user prompt simply if system prompt is provided
        final_prompt = prompt
        if system_prompt:
            final_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            completion_kwargs = self._completion_kwargs(kwargs)
        except ValueError as e:
            logger.error(f"Invalid inference parameters: {str(e)}")
            raise

        if LOG_PROMPT_CONTENT:
            logger.info(f"Starting raw generation with prompt: {prompt[:50]}...")
        else:
            logger.info(f"Starting raw generation ({len(prompt)} prompt chars)")
        start_time = time.time()

        # Send the raw prompt directly to the model
        response = self.llm.create_completion(
            prompt=final_prompt,
            stream=False,
            **completion_kwargs
        )

        duration = time.time() - start_time
        logger.info(f"Raw generation completed in {duration:.2f} seconds")

        if isinstance(response, dict) and "choices" in response and len(response["choices"]) > 0:
            raw_response = response["choices"][0]["text"]
        else:
            logger.warning(f"Unexpected response format: {type(response)}")
            raw_response = self._chunk_text(response) or str(response)

        return raw_response.strip()

    def generate_stream_raw(self, prompt: str, system_prompt: str = "", **kwargs) -> Iterator[Dict[str, Any]]:
        """Generate a streaming response using the raw prompt without any formatting."""
        if self.llm is None:
            raise ValueError("Model not loaded")

        # Combine system and user prompt simply if system prompt is provided
        final_prompt = prompt
        if system_prompt:
            final_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            completion_kwargs = self._completion_kwargs(kwargs)
        except ValueError as e:
            logger.error(f"Invalid inference parameters: {str(e)}")
            yield {"error": str(e)}
            return

        try:
            if LOG_PROMPT_CONTENT:
                logger.info(f"Starting raw streaming generation with prompt: {prompt[:50]}...")
            else:
                logger.info(f"Starting raw streaming generation ({len(prompt)} prompt chars)")

            stream = self.llm.create_completion(
                prompt=final_prompt,
                stream=True,
                **completion_kwargs
            )

            # Return each chunk as it comes in. The accumulated text is yielded
            # unmodified; stripping it per chunk makes trailing newlines flicker.
            collected_text = ""
            for chunk in stream:
                token = self._chunk_text(chunk)
                collected_text += token

                yield {
                    "token": token,
                    "text": collected_text
                }

        except Exception as e:
            logger.error(f"Error in raw streaming generation: {str(e)}")
            yield {
                "error": str(e)
            }

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        # _instance_lock (not the server's model_lock) is held across the native
        # call: bind self.llm to a local once and never re-read the attribute.
        with self._instance_lock:
            llm = self.llm
            if llm is None:
                # If no model is loaded, provide a rough estimate
                # Most models use roughly 3-4 characters per token on average
                return len(text) // 3

            try:
                # Use the model's tokenizer to get accurate count. add_bos=False
                # so this reports prompt tokens - otherwise every count is one
                # high and empty text costs 1 here but 0 on the estimate path.
                tokens = llm.tokenize(text.encode('utf-8'), add_bos=False)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Error counting tokens: {str(e)}, using rough estimate")
                # Fallback to rough estimate
                return len(text) // 3

    def get_max_context(self) -> int:
        """Get the effective context window for this model."""
        # Same lifetime guard as count_tokens - llama_n_ctx on a freed context
        # is a native access violation, not a catchable exception.
        with self._instance_lock:
            llm = self.llm
            if llm is not None:
                return llm.n_ctx()
        return self.loading_parameters.get("n_ctx", get_default_n_ctx(self.config.key))

    def get_context_usage(self, text: str) -> dict:
        """Get context usage information for any text (raw or formatted)."""
        token_count = self.count_tokens(text)
        max_context = self.get_max_context()

        usage_percentage = (token_count / max_context) * 100 if max_context > 0 else 0

        return {
            'token_count': token_count,
            'max_context': max_context,
            'usage_percentage': round(usage_percentage, 1),
            'remaining_tokens': max(0, max_context - token_count)
        }
