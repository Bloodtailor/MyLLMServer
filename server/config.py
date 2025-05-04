import os
from pathlib import Path

# Default generation settings
DEFAULT_N_GPU_LAYERS = -1  # -1 means use all available layers on GPU
DEFAULT_N_CTX = 2048       # Context window size

# Parsing settings
DEFAULT_MAX_RETRIES = 3

# Logging settings
LOG_LEVEL = "INFO"

# Model assignments for different use cases
MODEL_ASSIGNMENTS = {
    "MyMainLLM": {
        "name": "kunoichi",
        "model_path": "C:/Users/soulo/.cache/lm-studio/models/TheBloke/Kunoichi-7B-GGUF/kunoichi-7b.Q6_K.gguf",
        "inference_params": {
            "pre_prompt_prefix": "",
            "pre_prompt_suffix": "",
            "input_prefix": "\n### Instruction:\n",
            "input_suffix": "",
            "assistant_prefix": "\n### Response:\n",
            "assistant_suffix": ""
        },
        "default_params": {
            "temperature": 0.7,
            "max_tokens": 300,
            "top_p": 0.95,
        }
    },
    "MySecondLLM": {
        "name": "alphamonarch",
        "model_path": "C:/Users/soulo/.cache/lm-studio/models/mlabonne/AlphaMonarch-7B-GGUF/alphamonarch-7b.Q4_0.gguf",
        "inference_params": {
            "pre_prompt_prefix": "<|im_start|>system\n",
            "pre_prompt_suffix": "<|im_end|>\n",
            "input_prefix": "<|im_start|>user\n",
            "input_suffix": "<|im_end|>\n",
            "assistant_prefix": "<|im_start|>assistant\n",
            "assistant_suffix": "<|im_end|>\n"
        },
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