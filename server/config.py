import os
from pathlib import Path

# Default generation settings
DEFAULT_N_GPU_LAYERS = -1  # -1 means use all available layers on GPU
DEFAULT_N_CTX = 2048       # Context window size
DEFAULT_N_THREADS = 8      # Number of CPU threads to use for computation

# Global loading parameter definitions (apply to all models)
GLOBAL_LOADING_PARAMETERS = {
    "n_gpu_layers": {
        "default": DEFAULT_N_GPU_LAYERS,
        "min": -1,
        "max": 100,
        "type": "integer",
        "description": "Number of GPU layers (-1 for all available)"
    },
    "n_threads": {
        "default": DEFAULT_N_THREADS,
        "min": 1,
        "max": 32,
        "type": "integer",
        "description": "Number of CPU threads for computation"
    },
    "use_mlock": {
        "default": True,
        "type": "boolean",
        "description": "Keep model in memory (prevents swapping)"
    },
    "use_mmap": {
        "default": True,
        "type": "boolean",
        "description": "Use memory mapping for model files"
    },
    "f16_kv": {
        "default": True,
        "type": "boolean",
        "description": "Use half-precision for key-value cache"
    }
}

# Global inference parameter definitions (apply to all models unless overridden)
GLOBAL_INFERENCE_PARAMETERS = {
    "temperature": {
        "default": 0.7,
        "min": 0.0,
        "max": 2.0,
        "type": "float",
        "description": "Controls randomness in generation (0.0 = deterministic, 2.0 = very random)"
    },
    "max_tokens": {
        "default": 300,
        "min": 1,
        "max": 4096,
        "type": "integer",
        "description": "Maximum number of tokens to generate"
    },
    "top_p": {
        "default": 0.95,
        "min": 0.0,
        "max": 1.0,
        "type": "float",
        "description": "Nucleus sampling - cumulative probability cutoff"
    },
    "top_k": {
        "default": 40,
        "min": 1,
        "max": 100,
        "type": "integer",
        "description": "Top-k sampling - consider only top k tokens"
    },
    "repeat_penalty": {
        "default": 1.1,
        "min": 1.0,
        "max": 2.0,
        "type": "float",
        "description": "Penalty for repeating tokens (1.0 = no penalty)"
    },
    "min_p": {
        "default": 0.05,
        "min": 0.0,
        "max": 1.0,
        "type": "float",
        "description": "Minimum probability threshold for token selection"
    }
}

def validate_parameter(param_name, value, param_def):
    """Validate a parameter value against its definition."""
    param_type = param_def.get("type", "float")
    
    # Type conversion and validation
    try:
        if param_type == "boolean":
            if isinstance(value, str):
                value = value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        elif param_type == "integer":
            value = int(value)
        elif param_type == "float":
            value = float(value)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    except (ValueError, TypeError):
        raise ValueError(f"Invalid value for {param_name}: {value} (expected {param_type})")
    
    # Range validation
    if "min" in param_def and value < param_def["min"]:
        raise ValueError(f"{param_name} value {value} is below minimum {param_def['min']}")
    if "max" in param_def and value > param_def["max"]:
        raise ValueError(f"{param_name} value {value} is above maximum {param_def['max']}")
    
    return value

def get_loading_parameter_defaults():
    """Get default values for all loading parameters."""
    defaults = {}
    for param_name, param_def in GLOBAL_LOADING_PARAMETERS.items():
        defaults[param_name] = param_def["default"]
    return defaults

def get_inference_parameter_defaults(model_name=None):
    """Get default values for all inference parameters, with model-specific overrides."""
    defaults = {}
    
    # Start with global defaults
    for param_name, param_def in GLOBAL_INFERENCE_PARAMETERS.items():
        defaults[param_name] = param_def["default"]
    
    # Apply model-specific overrides if available
    if model_name and model_name in MODEL_ASSIGNMENTS:
        model_defaults = MODEL_ASSIGNMENTS[model_name].get("default_params", {})
        defaults.update(model_defaults)
    
    return defaults

# Model assignments for different use cases
MODEL_ASSIGNMENTS = {
    "MyMainLLM": {
        "name": "kunoichi",
        "model_path": "C:/Users/soulo/.cache/lm-studio/models/TheBloke/Kunoichi-7B-GGUF/kunoichi-7b.Q6_K.gguf",
        "max_context_window": 8192,  # Maximum context window supported by this model
        "inference_params": {
            "pre_prompt_prefix": "",
            "pre_prompt_suffix": "",
            "input_prefix": "\n### Instruction:\n",
            "input_suffix": "",
            "assistant_prefix": "\n### Response:\n",
            "assistant_suffix": ""
        },
        # Model-specific loading parameter overrides
        "loading_params": {
            "n_ctx": {
                "default": 2048,
                "min": 512,
                "max": 8192,
                "type": "integer",
                "description": "Context window size for this model"
            }
        },
        # Model-specific inference parameter overrides
        "default_params": {
            "temperature": 0.7,
            "max_tokens": 300,
            "top_p": 0.95,
        }
    },
    "MySecondLLM": {
        "name": "alphamonarch",
        "model_path": "C:/Users/soulo/.cache/lm-studio/models/mlabonne/AlphaMonarch-7B-GGUF/alphamonarch-7b.Q4_0.gguf",
        "max_context_window": 8192,  # Maximum context window supported by this model
        "inference_params": {
            "pre_prompt_prefix": "<|im_start|>system\n",
            "pre_prompt_suffix": "<|im_end|>\n",
            "input_prefix": "<|im_start|>user\n",
            "input_suffix": "<|im_end|>\n",
            "assistant_prefix": "<|im_start|>assistant\n",
            "assistant_suffix": "<|im_end|>\n"
        },
        # Model-specific loading parameter overrides
        "loading_params": {
            "n_ctx": {
                "default": 2048,
                "min": 512,
                "max": 8192,
                "type": "integer",
                "description": "Context window size for this model"
            }
        },
        # Model-specific inference parameter overrides
        "default_params": {
            "temperature": 0.8,
            "max_tokens": 300,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
        }
    }
}