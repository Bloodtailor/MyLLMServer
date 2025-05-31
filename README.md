# MyLLMServer - Local LLM Flask Server

A high-performance Flask server for running Large Language Models locally with CUDA acceleration, designed to work seamlessly with the companion Android application (https://github.com/Bloodtailor/my-llm-android-app.git).

## 🎯 Overview

MyLLMServer provides a REST API interface for interacting with local LLM models, enabling you to run powerful AI models on your PC and access them from mobile devices or other applications. The server handles model loading, memory management, and streaming responses efficiently.

## ✨ Features

### Core Functionality
- **Multiple Model Support**: Configure and switch between different GGUF models
- **Dynamic Model Loading**: Load/unload models on demand to manage memory
- **CUDA Acceleration**: Automatic GPU detection and utilization when available
- **Streaming Responses**: Real-time token streaming for immediate feedback
- **Context Management**: Configurable context windows and token counting
- **Raw Prompt Mode**: Send prompts exactly as typed without auto-formatting

### Performance & Reliability
- **Memory Optimization**: Smart model loading/unloading to prevent OOM errors
- **Connection Management**: Robust handling of multiple client connections
- **Error Handling**: Comprehensive error catching with detailed logging
- **System Monitoring**: Built-in performance and resource monitoring
- **Auto-setup**: Automated environment configuration with CUDA support

### Developer Features
- **RESTful API**: Clean, well-documented endpoints
- **Detailed Logging**: Rotating log files with configurable levels
- **Health Checks**: Server status and connectivity endpoints
- **Hot Configuration**: Model settings without server restart
- **Debug Tools**: GPU usage testing and diagnostics

## 🏗️ Architecture

```
MyLLMServer/
├── server.py              # Main Flask application
├── llm_manager.py          # LLM operations and model management
├── config.py               # Model configuration and parameters
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

Edit `config.py` to add your models:

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
            "repeat_penalty": 1.1
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

**POST `/model/load`**
```json
{
  "model": "MyMainLLM",
  "context_length": 4096
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
  "context_length": 4096
}
```

### Text Generation

**POST `/query`**
```json
{
  "prompt": "What is artificial intelligence?",
  "system_prompt": "You are a helpful assistant.",
  "model": "MyMainLLM",
  "stream": true
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

**Response**:
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
  "model_loaded": true,
  "memory_total": 34359738368,
  "gpu_available": true
}
```

**GET `/server/ping`**
```json
{
  "status": "online",
  "timestamp": "2025-05-30T10:30:00"
}
```

## ⚙️ Configuration

### Model Parameters

```python
# In config.py
DEFAULT_N_GPU_LAYERS = -1      # Use all GPU layers (-1) or specific count
DEFAULT_N_CTX = 2048           # Default context window
DEFAULT_N_THREADS = 8          # CPU threads for computation

MODEL_ASSIGNMENTS = {
    "YourModel": {
        "name": "display-name",
        "model_path": "/path/to/model.gguf",
        "max_context_window": 8192,
        "default_params": {
            "temperature": 0.7,     # Creativity (0.0-2.0)
            "max_tokens": 300,      # Response length limit
            "top_p": 0.95,         # Nucleus sampling
            "top_k": 40,           # Top-k sampling
            "repeat_penalty": 1.1   # Prevent repetition
        }
    }
}
```

### Server Settings

**Port Configuration**: Server runs on port 5000 by default
**CORS**: Enabled for all origins (configure in `server.py` for production)
**Logging**: Rotating logs in `logs/` directory (10MB per file, 5 backups)
**Timeouts**: Configurable connection and read timeouts


### Debug Mode

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

# Model status
curl http://localhost:5000/model/status
```


## 🤝 Integration

This server is designed to work with:
- **Android LLM App**: Primary mobile client
