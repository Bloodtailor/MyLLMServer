# MyLLMServer - Local LLM Flask Server

A high-performance Flask server for running Large Language Models locally with CUDA acceleration, designed to work seamlessly with the companion [Android application](https://github.com/Bloodtailor/my-llm-android-app.git).

## 🎯 Overview

MyLLMServer provides a REST API interface for interacting with local LLM models, enabling you to run powerful AI models on your PC and access them from mobile devices or other applications. The server handles model loading, memory management, and streaming responses efficiently.

## ✨ Features

### Core Functionality
- **Multiple Model Support**: Configure and switch between different GGUF models
- **Dynamic Model Loading**: Load/unload models on demand with custom parameters
- **CUDA Acceleration**: Automatic GPU detection and utilization when available
- **Streaming Responses**: Real-time token streaming for immediate feedback
- **Context Management**: Configurable context windows and token counting
- **Raw Prompt Mode**: Send prompts exactly as typed without auto-formatting

### Advanced Parameter Management
- **Loading Parameters**: Configure model loading with n_gpu_layers, n_ctx, n_threads, memory settings
- **Inference Parameters**: Real-time adjustment of temperature, top_p, top_k, repeat_penalty, min_p, max_tokens
- **Parameter Validation**: Server-side validation with min/max bounds and type checking
- **Model-Specific Defaults**: Each model can have its own parameter defaults and constraints

### Performance & Reliability
- **Memory Optimization**: Smart model loading/unloading to prevent OOM errors
- **Connection Management**: Robust handling of multiple client connections
- **Error Handling**: Comprehensive error catching with detailed logging
- **System Monitoring**: Built-in performance and resource monitoring
- **Auto-setup**: Automated environment configuration with CUDA support

### Developer Features
- **RESTful API**: Clean, well-documented endpoints with enhanced parameter support
- **Detailed Logging**: Rotating log files with configurable levels
- **Health Checks**: Server status and connectivity endpoints
- **Hot Configuration**: Model settings without server restart
- **Debug Tools**: GPU usage testing and diagnostics

## 🏗️ Architecture

```
MyLLMServer/
├── server.py              # Main Flask application with enhanced endpoints
├── llm_manager.py          # LLM operations, model management, and parameter handling
├── config.py               # Model configuration, parameters, and validation rules
├── setup_environment.py    # Automated environment setup
├── start_server.bat        # Windows startup script
├── requirements.txt        # Python dependencies
├── gpu_usage_test.py      # GPU diagnostic tool (optional)
├── logs/                  # Server log files (created automatically)
└── README.md
```

## 🚀 Quick Start

### 1. Environment Setup

**Automated Setup (Recommended)**:
```bash
git clone https://github.com/yourusername/MyLLMServer.git
cd MyLLMServer
python setup_environment.py
```

The setup script will:
- ✅ Check Python version (3.8+ required)
- ✅ Detect NVIDIA GPU and CUDA installation
- ✅ Verify Visual Studio Build Tools
- ✅ Create virtual environment
- ✅ Install dependencies with CUDA support
- ✅ Create log directories

**Manual Setup**:
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Configuration

Edit `config.py` to add your models with advanced parameter support:

```python
MODEL_ASSIGNMENTS = {
    "MyMainLLM": {
        "name": "kunoichi",
        "model_path": "C:/path/to/your/model.gguf",
        "max_context_window": 8192,
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
            "top_k": 40,
            "repeat_penalty": 1.1,
            "min_p": 0.05
        },
        "loading_params": {
            "custom_param": {
                "default": 128,
                "min": 1,
                "max": 512,
                "type": "integer",
                "description": "Custom model-specific parameter"
            }
        }
    }
}
```

### 3. Start Server

**Windows**:
```bash
start_server.bat
```

**Manual Start**:
```bash
venv\Scripts\activate
python server.py
```

The server will display your IP address - use this in your Android app settings.

## 📡 API Reference

### Model Management

**GET `/models`**
```json
{
  "models": ["MyMainLLM", "MySecondLLM"]
}
```

**POST `/model/load`** (Enhanced with loading parameters)
```json
{
  "model": "MyMainLLM",
  "n_ctx": 4096,
  "n_gpu_layers": -1,
  "n_threads": 8,
  "use_mlock": true,
  "use_mmap": true,
  "f16_kv": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Model MyMainLLM loaded successfully",
  "model": "MyMainLLM",
  "loading_parameters": {
    "n_ctx": 4096,
    "n_gpu_layers": -1,
    "n_threads": 8,
    "use_mlock": true,
    "use_mmap": true,
    "f16_kv": true
  }
}
```

**POST `/model/unload`**
```json
{
  "status": "success",
  "message": "Model unloaded successfully"
}
```

**GET `/model/status`**
```json
{
  "loaded": true,
  "current_model": "MyMainLLM",
  "context_length": 4096,
  "loading_parameters": {
    "n_ctx": 4096,
    "n_gpu_layers": -1,
    "n_threads": 8
  }
}
```

### Parameter Management

**GET `/model/loading-parameters`**
```json
{
  "global_defaults": {
    "n_ctx": {
      "default": 2048,
      "min": 128,
      "max": 32768,
      "type": "integer",
      "description": "Context window size"
    },
    "n_gpu_layers": {
      "default": -1,
      "min": -1,
      "max": 100,
      "type": "integer",
      "description": "Number of layers to offload to GPU (-1 for all)"
    }
  },
  "model_specific": {
    "MyMainLLM": {
      "custom_param": {
        "default": 128,
        "min": 1,
        "max": 512,
        "type": "integer",
        "description": "Custom model parameter"
      }
    }
  }
}
```

**GET `/model/inference-parameters`**
```json
{
  "model": "MyMainLLM",
  "parameters": {
    "temperature": {
      "current": 0.7,
      "default": 0.7,
      "min": 0.0,
      "max": 2.0,
      "type": "float",
      "description": "Controls randomness in generation"
    },
    "max_tokens": {
      "current": 300,
      "default": 300,
      "min": 1,
      "max": 2048,
      "type": "integer",
      "description": "Maximum tokens to generate"
    }
  }
}
```

### Text Generation

**POST `/query`** (Enhanced with inference parameters)
```json
{
  "prompt": "What is artificial intelligence?",
  "system_prompt": "You are a helpful assistant.",
  "model": "MyMainLLM",
  "stream": true,
  "temperature": 0.8,
  "max_tokens": 500,
  "top_p": 0.9,
  "top_k": 50,
  "repeat_penalty": 1.2,
  "min_p": 0.1
}
```

**Streaming Response** (NDJSON):
```json
{"status": "processing", "partial": ""}
{"status": "generating", "partial": "Artificial intelligence"}
{"status": "generating", "partial": "Artificial intelligence is..."}
{"status": "complete", "response": "Full response text"}
```

### Utilities

**POST `/count_tokens`**
```json
{
  "text": "Your text here",
  "model": "MyMainLLM"
}
```

**Response:**
```json
{
  "text": "Your text here",
  "model": "MyMainLLM",
  "context_usage": {
    "token_count": 156,
    "max_context": 4096,
    "usage_percentage": 3.8,
    "remaining_tokens": 3940
  }
}
```

**GET `/server/info`**
```json
{
  "server_platform": "Windows-10",
  "python_version": "3.11.0",
  "current_model": "MyMainLLM",
  "loading_parameters": {
    "n_ctx": 4096,
    "n_gpu_layers": -1
  },
  "model_loaded": true,
  "memory_total": 34359738368,
  "gpu_available": true
}
```

**GET `/server/ping`**
```json
{
  "status": "online",
  "timestamp": "2025-06-05T10:30:00"
}
```

**GET `/model/parameters`** (Model prefix/suffix parameters)
```json
{
  "model": "MyMainLLM",
  "pre_prompt_prefix": "",
  "pre_prompt_suffix": "",
  "input_prefix": "\n### Instruction:\n",
  "input_suffix": "",
  "assistant_prefix": "\n### Response:\n",
  "assistant_suffix": ""
}
```

## ⚙️ Configuration

### Global Loading Parameters

Configure in `config.py`:

```python
GLOBAL_LOADING_PARAMETERS = {
    "n_ctx": {
        "default": 2048,
        "min": 128,
        "max": 32768,
        "type": "integer",
        "description": "Context window size"
    },
    "n_gpu_layers": {
        "default": -1,
        "min": -1,
        "max": 100,
        "type": "integer", 
        "description": "GPU layers (-1 for all)"
    },
    "n_threads": {
        "default": 8,
        "min": 1,
        "max": 32,
        "type": "integer",
        "description": "CPU threads"
    },
    "use_mlock": {
        "default": True,
        "type": "boolean",
        "description": "Use memory locking"
    },
    "use_mmap": {
        "default": True,
        "type": "boolean", 
        "description": "Use memory mapping"
    },
    "f16_kv": {
        "default": True,
        "type": "boolean",
        "description": "Use 16-bit key-value cache"
    }
}
```

### Global Inference Parameters

```python
GLOBAL_INFERENCE_PARAMETERS = {
    "temperature": {
        "default": 0.7,
        "min": 0.0,
        "max": 2.0,
        "type": "float",
        "description": "Controls randomness"
    },
    "max_tokens": {
        "default": 300,
        "min": 1,
        "max": 2048,
        "type": "integer",
        "description": "Maximum tokens to generate"
    },
    "top_p": {
        "default": 0.95,
        "min": 0.0,
        "max": 1.0,
        "type": "float",
        "description": "Nucleus sampling threshold"
    },
    "top_k": {
        "default": 40,
        "min": 0,
        "max": 200,
        "type": "integer",
        "description": "Top-k sampling limit"
    },
    "repeat_penalty": {
        "default": 1.1,
        "min": 0.1,
        "max": 2.0,
        "type": "float",
        "description": "Repetition penalty"
    },
    "min_p": {
        "default": 0.05,
        "min": 0.0,
        "max": 1.0,
        "type": "float",
        "description": "Minimum probability threshold"
    }
}
```

### Model-Specific Configuration

Each model can override defaults and add custom parameters:

```python
MODEL_ASSIGNMENTS = {
    "YourModel": {
        "name": "display-name",
        "model_path": "/path/to/model.gguf",
        "max_context_window": 8192,
        "default_params": {
            "temperature": 0.8,  # Override global default
            "max_tokens": 500
        },
        "loading_params": {
            "custom_layer_count": {
                "default": 32,
                "min": 1,
                "max": 64,
                "type": "integer",
                "description": "Custom layer parameter"
            }
        }
    }
}
```

### Server Settings

**Port Configuration**: Server runs on port 5000 by default
**CORS**: Enabled for all origins (configure in `server.py` for production)
**Logging**: Rotating logs in `logs/` directory (10MB per file, 5 backups)
**Timeouts**: Configurable connection and read timeouts

## 🔧 Parameter Management

### Loading Parameters
Control how models are loaded into memory:
- **n_ctx**: Context window size (128-32768)
- **n_gpu_layers**: GPU layer count (-1 for all available)
- **n_threads**: CPU thread count (1-32)
- **use_mlock**: Memory locking for performance
- **use_mmap**: Memory mapping for efficiency
- **f16_kv**: 16-bit key-value cache

### Inference Parameters
Control text generation behavior:
- **temperature**: Randomness (0.0-2.0)
- **max_tokens**: Response length limit (1-2048)
- **top_p**: Nucleus sampling (0.0-1.0)
- **top_k**: Top-k sampling (0-200)
- **repeat_penalty**: Prevent repetition (0.1-2.0)
- **min_p**: Minimum probability threshold (0.0-1.0)

### Parameter Validation
All parameters are validated server-side with:
- Type checking (integer, float, boolean)
- Range validation (min/max bounds)
- Error reporting with specific details
- Automatic fallback to defaults

## 🐛 Debug Mode

Enable detailed logging:
```python
# In server.py, change logging level
logger.setLevel(logging.DEBUG)
```

Run with debug output:
```bash
python server.py --debug
```

## 📊 Monitoring

### Health Checks
```bash
# Quick health check
curl http://localhost:5000/server/ping

# Detailed system info
curl http://localhost:5000/server/info

# Model status with loading parameters
curl http://localhost:5000/model/status

# Available loading parameters
curl http://localhost:5000/model/loading-parameters

# Current inference parameters
curl http://localhost:5000/model/inference-parameters
```

### Performance Testing
```bash
# Test parameter validation
curl -X POST http://localhost:5000/model/load \
  -H "Content-Type: application/json" \
  -d '{"model": "MyMainLLM", "n_ctx": 4096, "temperature": 0.8}'

# Test inference with custom parameters
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "model": "MyMainLLM", "temperature": 0.8, "max_tokens": 100}'
```

## 🔄 Recent Updates

### Version 2.0 Features
- **Enhanced Parameter Management**: Full loading and inference parameter control
- **Real-time Parameter Validation**: Server-side validation with detailed error reporting
- **Model-Specific Defaults**: Each model can have custom parameter configurations
- **Raw Prompt Processing**: Direct prompt handling without auto-formatting
- **Improved Error Handling**: Better error messages and recovery
- **Extended API**: New endpoints for parameter discovery and management

### Breaking Changes
- Loading parameters now validated and structured
- Inference parameters sent directly in `/query` endpoint
- Model loading requires explicit parameter specification
- Enhanced response formats with parameter information

## 🤝 Integration

This server is designed to work with:
- **Android LLM App**: Primary mobile client with full parameter control
- **Web Interfaces**: Any HTTP client supporting the REST API
- **Custom Applications**: Full API access for integration

## 🛠️ Troubleshooting

### Common Issues

**Parameter Validation Errors**:
```bash
# Check available parameters first
curl http://localhost:5000/model/loading-parameters
curl http://localhost:5000/model/inference-parameters
```

**Model Loading Failures**:
- Check VRAM availability for large models
- Verify model path in `config.py`
- Review loading parameters (especially `n_gpu_layers` and `n_ctx`)

**Performance Issues**:
- Adjust `n_threads` for your CPU
- Optimize `n_gpu_layers` based on VRAM
- Monitor memory usage during operation

### GPU Issues

If models aren't loading to GPU:
1. Check CUDA installation
2. Verify GPU VRAM availability
3. Set `n_gpu_layers` to appropriate value
4. Check server logs for GPU detection

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🔗 Related Projects

- **[Android LLM App](https://github.com/yourusername/my-llm-android-app)** - Mobile client with full parameter control
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** - Core LLM inference library

---

**Note**: This server requires proper model configuration and adequate system resources. See the setup instructions for detailed requirements and configuration steps.