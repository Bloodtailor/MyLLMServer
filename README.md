# MyLLMServer - Local LLM Flask Server

A small Flask server for running GGUF language models locally with CUDA acceleration, designed to work with the companion [Android application](https://github.com/Bloodtailor/my-llm-android-app).

## 🎯 Overview

MyLLMServer wraps [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) in a REST API so the model running on your PC can be reached from your phone or any other client on your home network. The server handles model loading, parameter validation, token counting and streaming responses.

> ⚠️ **Home-network project.** The server binds to `0.0.0.0:5000` (every network interface) and has **no authentication, no TLS and no rate limiting**. Anyone who can reach that port can load models and generate text on your machine. Run it on a trusted home LAN only — never forward the port through your router or expose it to the internet.

## ✨ Features

### Core Functionality
- **Multiple Model Support**: Configure and switch between different GGUF models
- **Dynamic Model Loading**: Load/unload models on demand with custom loading parameters
- **CUDA Acceleration**: Setup script installs a GPU-capable build and verifies GPU offload actually works
- **Streaming Responses**: Real-time token streaming over a simple NDJSON protocol
- **Token Counting**: Live context-usage estimates, with or without a model loaded
- **Raw Prompt Mode**: Prompts go to the model exactly as typed — the server applies no chat template

### Parameter Management
- **Loading Parameters**: `n_gpu_layers`, `n_ctx`, `n_threads`, `use_mlock`, `use_mmap`
- **Inference Parameters**: `temperature`, `max_tokens`, `top_p`, `top_k`, `repeat_penalty`, `min_p`
- **Parameter Validation**: Type checking and min/max bounds, with `400` responses that name the offending value
- **Model-Specific Defaults**: Each model can override the global inference defaults and narrow its own loading bounds
- **Discoverable**: Endpoints that hand a client the full parameter definitions so it can build a settings screen

### Reliability
- **One Model at a Time**: A single lock serializes loading, unloading and generation, so overlapping requests can't corrupt the llama-cpp context
- **Clean Unloading**: `Llama.close()` is called before the reference is dropped, so VRAM is actually released
- **No Stale State**: A failed load leaves the server reporting "no model loaded" rather than lying about the previous one
- **Honest Errors**: Generation failures surface as HTTP `500` with an `error` key instead of being smuggled into the response text
- **Rotating Logs**: One console handler plus a rotating file handler, written next to the server regardless of working directory

## 🏗️ Architecture

All of the Python code lives in the **`server/`** subfolder — every command in this README is run from inside it.

```
MyLLMServer/
├── server/                    # ← all commands are run from here
│   ├── server.py              # Flask app: endpoints, logging, model lock
│   ├── llm_manager.py         # llama-cpp wrapper: load/close, raw generation, tokenizing
│   ├── config.py              # Host/port, model assignments, parameter definitions + validation
│   ├── setup_environment.py   # One-shot environment setup (venv + CUDA-aware install)
│   ├── start_server.bat       # Windows launcher (uses the venv, works from any folder)
│   ├── requirements.txt       # Hand-curated pins (CPU-only llama-cpp-python build)
│   ├── requirements.lock.txt  # pip freeze snapshot, written by setup_environment.py
│   ├── logs/                  # Rotating server logs (created automatically)
│   └── venv/                  # Virtual environment (created by setup_environment.py)
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### 1. Environment Setup

**Automated Setup (Recommended)**:
```bash
git clone https://github.com/Bloodtailor/MyLLMServer.git
cd MyLLMServer/server
python setup_environment.py
```

The setup script will:
- ✅ Check Python version (3.8+ required, 3.11 recommended)
- ✅ Detect NVIDIA GPU (`nvidia-smi`) and CUDA Toolkit (`CUDA_PATH` / `nvcc`)
- ✅ Look for Visual Studio Build Tools (only needed if it has to compile)
- ✅ Create `server/venv/`
- ✅ Install `flask`, `flask-cors`, `psutil`, then `llama-cpp-python` with GPU support
- ✅ Report whether the installed build can *actually* offload to the GPU
- ✅ Check that the model paths in `config.py` exist
- ✅ Create `server/logs/` and write `requirements.lock.txt`

If some checks fail it asks whether to continue, then falls back gracefully.

**How it installs `llama-cpp-python`** (in this order — the first one that works wins):

1. **Prebuilt CUDA wheel** — no compiler needed, takes seconds:
   ```bash
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
   ```
2. **Source build with CUDA** — needs the CUDA Toolkit and VS Build Tools, takes 5–10 minutes:
   ```bash
   set CMAKE_ARGS=-DGGML_CUDA=on
   set FORCE_CMAKE=1
   pip install llama-cpp-python --force-reinstall --no-cache-dir --no-binary llama-cpp-python
   ```
   `--no-binary` matters: without it pip quietly reuses a prebuilt CPU wheel and cmake never runs.
3. **CPU-only fallback** — plain `pip install llama-cpp-python`. Everything still works, just slowly.

After every path the script runs this check, so a silent CPU fallback can't hide from you:
```bash
python -c "import llama_cpp; print(bool(llama_cpp.llama_supports_gpu_offload()))"
```

**Manual Setup**:
```bash
cd server
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux/Mac

pip install -r requirements.txt
```

⚠️ `requirements.txt` pins the plain PyPI build of `llama-cpp-python`, which is **CPU-only**. If you want GPU offload, reinstall it from the CUDA wheel index (method 1 above) or just run `setup_environment.py`.

### 2. Model Configuration

Edit `server/config.py` and point `MODEL_ASSIGNMENTS` at your `.gguf` files. Each entry's keys map onto the `ModelConfig` dataclass, so `name`, `model_path`, `inference_params` and `default_params` all have to be present:

```python
MODEL_ASSIGNMENTS = {
    "MyMainLLM": {                      # the key is the name the API uses
        "name": "kunoichi",             # friendly label, not used for lookups
        "model_path": "C:/path/to/your/model.gguf",
        "max_context_window": 8192,     # documentation only - see note below
        "inference_params": {           # prompt scaffolding, served to clients
            "pre_prompt_prefix": "",
            "pre_prompt_suffix": "",
            "input_prefix": "\n### Instruction:\n",
            "input_suffix": "",
            "assistant_prefix": "\n### Response:\n",
            "assistant_suffix": ""
        },
        "default_params": {             # overrides the global inference defaults
            "temperature": 0.7,
            "max_tokens": 300,
            "top_p": 0.95
        },
        "loading_params": {             # overrides/adds loading parameter definitions
            "n_ctx": {
                "default": 2048,
                "min": 512,
                "max": 8192,
                "type": "integer",
                "description": "Context window size for this model"
            }
        }
    }
}
```

Three things worth knowing:

- **`n_ctx` is per-model.** It is not in the global loading parameters — it only exists as a `loading_params` entry on each model, which is why both shipped models declare it. A model without one falls back to `DEFAULT_N_CTX` (2048).
- **`max_context_window` is declarative.** Nothing enforces it; the `n_ctx` bounds under `loading_params` are what the server actually validates against. Keep the two consistent by hand.
- **`inference_params` are served, not applied.** `/query` sends your prompt through untouched. These prefixes/suffixes exist so the client (the Android app) can build the prompt itself and show you exactly what the model will see.

### 3. Start Server

**Windows**:
```bash
cd server
start_server.bat
```

`start_server.bat` switches to its own folder first, so you can also double-click it from anywhere. It runs `venv\Scripts\python.exe` directly (no activation needed), installs `requirements.txt` if the venv looks empty, and stops with a pointer to `setup_environment.py` if there's no venv at all.

**Manual Start**:
```bash
cd server
venv\Scripts\activate
python server.py
```

Either way the console prints your machine's IPv4 address — that's what you enter in the Android app's settings.

## 📡 API Reference

Base URL: `http://<your-lan-ip>:5000`

All POST bodies are parsed with `force=True`, so the `Content-Type` header is ignored — anything that parses as JSON is accepted. Unless a specific status is listed below, unexpected failures return `500` with `{"error": "<message>"}`.

### Model Management

**GET `/models`** — the keys of `MODEL_ASSIGNMENTS` (not the friendly `name` fields)
```json
{
  "models": ["MyMainLLM", "MySecondLLM"]
}
```

**POST `/model/load`** — load a model, optionally overriding its loading parameters
```json
{
  "model": "MyMainLLM",
  "n_ctx": 4096,
  "n_gpu_layers": -1,
  "n_threads": 8,
  "use_mlock": true,
  "use_mmap": true
}
```

Every key except `model` is optional; anything omitted falls back to that model's defaults, and unknown keys are ignored. Loading unloads the previously loaded model first — unless the request matches what's already resident, in which case it's a no-op that still reports success.

**Response `200`:**
```json
{
  "status": "success",
  "message": "Model MyMainLLM loaded successfully",
  "model": "MyMainLLM",
  "loading_parameters": {
    "n_gpu_layers": -1,
    "n_threads": 8,
    "use_mlock": true,
    "use_mmap": true,
    "n_ctx": 4096
  }
}
```

`loading_parameters` is the **full effective set** the model was loaded with, not just the keys you sent.

**Error responses:**
```json
// 400 - unknown model name
{"error": "Unknown model: NotAModel"}

// 400 - a parameter is out of bounds or the wrong type
{
  "error": "Invalid loading parameters: n_ctx value 99999 is above maximum 8192",
  "invalid_parameters": ["n_ctx value 99999 is above maximum 8192"]
}

// 500 - the request was fine but the load failed
{"error": "Model path does not exist: C:/.../kunoichi-7b.Q6_K.gguf"}
```

Only parameter validation produces a `400`; a llama-cpp failure (a missing `.gguf`, out of VRAM) is a server fault and returns `500`.

**POST `/model/unload`** — body is ignored
```json
{"status": "success", "message": "Model MyMainLLM unloaded successfully"}
```
```json
{"status": "success", "message": "No model was loaded"}
```

This blocks until any in-flight generation finishes, then calls `Llama.close()` so VRAM is released.

**GET `/model/status`**
```json
{
  "loaded": true,
  "current_model": "MyMainLLM",
  "context_length": 4096,
  "loading_parameters": {
    "n_gpu_layers": -1,
    "n_threads": 8,
    "use_mlock": true,
    "use_mmap": true,
    "n_ctx": 4096
  }
}
```

`context_length` comes from the live model (`llm.n_ctx()`), and `loading_parameters` is the full effective set — both are correct after a lazy load through `/query`, not just after an explicit `/model/load`. With nothing loaded:
```json
{"loaded": false, "current_model": null, "context_length": null, "loading_parameters": null}
```

### Parameter Discovery

**GET `/model/loading-parameters`** — parameter *definitions*, not current values
```json
{
  "global_defaults": {
    "n_gpu_layers": {
      "default": -1, "min": -1, "max": 100, "type": "integer",
      "description": "Number of GPU layers (-1 for all available)"
    },
    "n_threads": {
      "default": 8, "min": 1, "max": 32, "type": "integer",
      "description": "Number of CPU threads for computation"
    },
    "use_mlock": {
      "default": true, "type": "boolean",
      "description": "Keep model in memory (prevents swapping)"
    },
    "use_mmap": {
      "default": true, "type": "boolean",
      "description": "Use memory mapping for model files"
    }
  },
  "model_specific": {
    "MyMainLLM": {
      "n_ctx": {
        "default": 2048, "min": 512, "max": 8192, "type": "integer",
        "description": "Context window size for this model"
      }
    },
    "MySecondLLM": {
      "n_ctx": {
        "default": 2048, "min": 512, "max": 8192, "type": "integer",
        "description": "Context window size for this model"
      }
    }
  }
}
```

A client building a settings screen needs to **merge** `global_defaults` with the `model_specific` entry for the selected model — `n_ctx` only appears in the latter.

**GET `/model/inference-parameters?model=MySecondLLM`** — `model` is optional and defaults to the loaded model, then to `MyMainLLM`
```json
{
  "model": "MySecondLLM",
  "parameters": {
    "temperature": {
      "current": 0.8, "default": 0.7, "min": 0.0, "max": 2.0, "type": "float",
      "description": "Controls randomness in generation (0.0 = deterministic, 2.0 = very random)"
    },
    "max_tokens": {
      "current": 300, "default": 300, "min": 1, "max": 4096, "type": "integer",
      "description": "Maximum number of tokens to generate"
    }
  }
}
```
(abridged — all six inference parameters are returned)

`default` is the global default and `current` is that default with the model's `default_params` applied on top. Neither reflects what the last request actually used — the server keeps no session state. An unknown `model` returns `400 {"error": "Unknown model: X"}`.

**GET `/model/parameters?model=MyMainLLM`** — the model's prompt prefixes/suffixes
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

Read-only. The server never applies these itself; they're here so the client can assemble the prompt. Unknown `model` → `400`.

### Text Generation

**POST `/query`**
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

| Key | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | Required; empty → `400` |
| `system_prompt` | string | `""` | Joined as `"{system}\n\n{prompt}"`. No chat template is applied |
| `model` | string | loaded model, else `MyMainLLM` | Loaded on demand if not already resident |
| `stream` | boolean | `true` | Any truthy value streams |
| `temperature` | float | 0.7 | 0.0 – 2.0 |
| `max_tokens` | integer | 300 | 1 – 4096 |
| `top_p` | float | 0.95 | 0.0 – 1.0 |
| `top_k` | integer | 40 | 1 – 100 |
| `repeat_penalty` | float | 1.1 | 1.0 – 2.0 |
| `min_p` | float | 0.05 | 0.0 – 1.0 |

Any other key is ignored (there is no `stop` sequence support). Omitted parameters fall back to the model's defaults.

**Pre-flight errors** — returned as real HTTP errors *before* any streaming starts:
```json
// 400 - no prompt
{"error": "No prompt provided"}

// 400 - unknown model
{"error": "Unknown model: NotAModel"}

// 400 - parameter out of bounds (checked before the prompt check)
{
  "error": "Invalid inference parameters: temperature value 5.0 is above maximum 2.0",
  "invalid_parameters": ["temperature value 5.0 is above maximum 2.0"]
}
```

#### Streaming protocol (NDJSON)

`Content-Type: application/x-ndjson`, HTTP/1.1 chunked, status always `200`. This is **not** SSE — there are no `data:` prefixes, no blank-line separators and no event names. One JSON object per `\n`-terminated line:

```
{"status":"processing","partial":""}
{"status":"generating","partial":"Artificial"}
{"status":"generating","partial":"Artificial intelligence"}
{"status":"generating","partial":"Artificial intelligence is"}
{"status":"complete","response":"Artificial intelligence is..."}
```

Protocol details a client must get right:

- **`partial` is cumulative, not a delta.** Each line carries the entire response so far — replace your buffer, never append.
- **The `processing` line arrives first, before the model is loaded.** It's sent before the server takes the model lock, so a cold start still gives you bytes immediately instead of a silent 30-second wait.
- **`partial` is never trimmed.** Leading and trailing whitespace is preserved exactly as the model produced it, so text doesn't flicker as newlines come and go.
- **The terminal line uses `response`, not `partial`,** and it *is* stripped — that single `.strip()` is the only text massaging the server does.
- **Errors mid-stream** replace the `complete` line and end the stream:
  ```
  {"status":"error","error":"<message>"}
  ```
- **A dropped connection just stops.** There's no terminal line, and nothing checks for client disconnect — the model keeps generating to `max_tokens`. There is no cancel endpoint.

#### Non-streaming (`"stream": false`)

```json
// 200
{"response": "Artificial intelligence is..."}

// 500 - generation failed
{"error": "<message>"}
```

`response` only ever contains real model output; failures are never disguised as text.

### Utilities

**POST `/count_tokens`**
```json
{
  "text": "Your text here",
  "model": "MyMainLLM"
}
```

**Response `200`:**
```json
{
  "text": "Your text here",
  "model": "MyMainLLM",
  "context_usage": {
    "token_count": 4,
    "max_context": 2048,
    "usage_percentage": 0.2,
    "remaining_tokens": 2044
  }
}
```

- With the requested model loaded, `token_count` uses the real tokenizer (without a BOS token, so it counts prompt tokens) and `max_context` is the live `n_ctx`.
- With nothing loaded, `token_count` is a `len(text) // 3` estimate and `max_context` is the `n_ctx` that model *would* be loaded with — so the usage meter doesn't jump when a model loads.
- `remaining_tokens` never goes negative.
- Empty `text` is valid and returns a normal response, counting `0` tokens on both paths. A non-string `text` → `400 {"error": "text must be a string"}`. Unknown `model` → `400`.
- This is the one model-touching endpoint that does **not** take the model lock (tokenizing only reads the vocabulary), so the app can call it on every keystroke while a generation is running. A separate short-held instance lock still keeps it from overlapping an unload freeing the model.

**GET `/server/info`**
```json
{
  "server_platform": "Windows-10-10.0.26200-SP0",
  "python_version": "3.10.10",
  "current_model": "MyMainLLM",
  "loading_parameters": {"n_gpu_layers": -1, "n_threads": 8, "use_mlock": true, "use_mmap": true, "n_ctx": 2048},
  "model_loaded": true,
  "memory_total": 68325322752,
  "memory_available": 22628827136,
  "memory_percent": 66.9,
  "disk_total": 994627096576,
  "disk_free": 179794038784,
  "disk_percent": 81.9,
  "cpu_percent": 0.0,
  "cpu_count": 16
}
```

Disk figures are for the drive the server lives on. `cpu_percent` is sampled non-blocking against a baseline taken at startup. If `psutil` is missing — or fails to read the system — that whole block is replaced by a `note` field and the rest of the response is still returned.

**GET `/server/ping`**
```json
{
  "status": "online",
  "timestamp": "2026-08-12T04:02:42.212566"
}
```

Local naive time, no timezone. Returns `online` whether or not a model is loaded.

**Unknown endpoints** get a `404` listing the real routes, generated from Flask's URL map:
```json
{
  "error": "Endpoint not found",
  "available_endpoints": [
    "/count_tokens", "/model/inference-parameters", "/model/load",
    "/model/loading-parameters", "/model/parameters", "/model/status",
    "/model/unload", "/models", "/query", "/server/info", "/server/ping"
  ]
}
```

## ⚙️ Configuration

### Network Settings

```python
# config.py
SERVER_HOST = "0.0.0.0"    # Listen on all interfaces so the phone can reach it
SERVER_PORT = 5000
```

### Global Loading Parameters

These apply to every model. Note there is **no global `n_ctx`** — context size is declared per model (see below).

```python
GLOBAL_LOADING_PARAMETERS = {
    "n_gpu_layers": {
        "default": -1,
        "min": -1,
        "max": 100,
        "type": "integer",
        "description": "Number of GPU layers (-1 for all available)"
    },
    "n_threads": {
        "default": 8,
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
    }
}
```

### Global Inference Parameters

```python
GLOBAL_INFERENCE_PARAMETERS = {
    "temperature": {
        "default": 0.7, "min": 0.0, "max": 2.0, "type": "float",
        "description": "Controls randomness in generation"
    },
    "max_tokens": {
        "default": 300, "min": 1, "max": 4096, "type": "integer",
        "description": "Maximum number of tokens to generate"
    },
    "top_p": {
        "default": 0.95, "min": 0.0, "max": 1.0, "type": "float",
        "description": "Nucleus sampling - cumulative probability cutoff"
    },
    "top_k": {
        "default": 40, "min": 1, "max": 100, "type": "integer",
        "description": "Top-k sampling - consider only top k tokens"
    },
    "repeat_penalty": {
        "default": 1.1, "min": 1.0, "max": 2.0, "type": "float",
        "description": "Penalty for repeating tokens (1.0 = no penalty)"
    },
    "min_p": {
        "default": 0.05, "min": 0.0, "max": 1.0, "type": "float",
        "description": "Minimum probability threshold for token selection"
    }
}
```

### Server Settings

- **Host/Port**: `0.0.0.0:5000`, from `SERVER_HOST` / `SERVER_PORT` in `config.py`
- **Debug**: always off. Because the server listens on every interface, Werkzeug's debugger would be an interactive shell for the whole LAN
- **Threading**: `threaded=True`, but a single reentrant lock serializes model load, unload and generation — one generation at a time, queued rather than interleaved
- **CORS**: enabled for all origins (`flask-cors`)
- **Logging**: console + rotating file, `server/logs/llm_server_YYYYMMDD.log`, 10 MB per file, 5 backups. The path is anchored to `server.py`, so logs land in the same place no matter where you launched from

## 🔧 Parameter Management

### Loading Parameters
Control how a model is loaded into memory:

| Parameter | Type | Default | Range | Scope |
|---|---|---|---|---|
| `n_gpu_layers` | integer | -1 | -1 – 100 | global |
| `n_threads` | integer | 8 | 1 – 32 | global |
| `use_mlock` | boolean | true | — | global |
| `use_mmap` | boolean | true | — | global |
| `n_ctx` | integer | 2048 | 512 – 8192 | per-model (both shipped models) |

### Inference Parameters
Control generation behaviour:

| Parameter | Type | Default | Range |
|---|---|---|---|
| `temperature` | float | 0.7 | 0.0 – 2.0 |
| `max_tokens` | integer | 300 | 1 – 4096 |
| `top_p` | float | 0.95 | 0.0 – 1.0 |
| `top_k` | integer | 40 | 1 – 100 |
| `repeat_penalty` | float | 1.1 | 1.0 – 2.0 |
| `min_p` | float | 0.05 | 0.0 – 1.0 |

A model's `default_params` override the defaults above — `MySecondLLM`, for example, reports `temperature: 0.8`.

### Parameter Validation
Every parameter that reaches `/query` or `/model/load` is validated server-side:
- **Type coercion**: `"0.8"` becomes `0.8`; booleans accept `true`, `"true"`, `"1"`, `"yes"`, `"on"`
- **Range checking** against the min/max in the definition
- **Rejected, not ignored**: a bad value returns `400` with a message naming the parameter, the value and the bound it broke, plus an `invalid_parameters` list
- **Unknown keys are dropped** silently — only parameters the server advertises are read

## 🐛 Logging & Debugging

There is no `--debug` flag, and Flask's debug mode is deliberately never enabled (see Server Settings). To get more detail, raise the log level in `server.py`:

```python
# in setup_logging()
root.setLevel(logging.DEBUG)
```

`llama-cpp` itself is loaded with `verbose=False`; flip that in `llm_manager.py` if you want its loading diagnostics on the console.

Logs go to both the console and `server/logs/llm_server_YYYYMMDD.log`. Every module logs through the root logger, so a line appears exactly once in each place.

## 🔒 Privacy

Prompt text is **never written to the log files by default** — the server logs only character counts (e.g. `Received prompt (245 chars)`), so your conversations don't accumulate in `server/logs/`. If you need to see prompt text while debugging, set `LOG_PROMPT_CONTENT = True` in `config.py` and set it back when you're done; the logs stay on this machine either way.

The `logs/` directory (and any `*.log`, `*.db`, or `*.sqlite*` file) is gitignored at both the repo root and inside `server/`, so user data cannot be committed to the repository — nothing user-generated has ever been part of this repo's history. Responses are never logged at all, only their lengths.

## 📊 Monitoring

### Health Checks
```bash
# Quick health check
curl http://localhost:5000/server/ping

# Detailed system info
curl http://localhost:5000/server/info

# Model status with effective loading parameters
curl http://localhost:5000/model/status

# Available loading parameters (definitions)
curl http://localhost:5000/model/loading-parameters

# Inference parameters for a specific model
curl "http://localhost:5000/model/inference-parameters?model=MySecondLLM"
```

### Trying the API
```bash
# Load a model with explicit parameters
curl -X POST http://localhost:5000/model/load -H "Content-Type: application/json" -d "{\"model\":\"MyMainLLM\",\"n_ctx\":4096,\"n_gpu_layers\":-1}"

# Watch validation reject a bad value (400)
curl -X POST http://localhost:5000/model/load -H "Content-Type: application/json" -d "{\"model\":\"MyMainLLM\",\"n_ctx\":99999}"

# Non-streaming generation
curl -X POST http://localhost:5000/query -H "Content-Type: application/json" -d "{\"prompt\":\"Hello!\",\"stream\":false,\"temperature\":0.8,\"max_tokens\":100}"

# Streaming generation (one JSON object per line)
curl -N -X POST http://localhost:5000/query -H "Content-Type: application/json" -d "{\"prompt\":\"Hello!\",\"max_tokens\":50}"

# Free the VRAM again
curl -X POST http://localhost:5000/model/unload
```

## 🔄 Recent Updates

### Reliability pass
- **Concurrency**: model load, unload and generation now run under one lock, so two phones hitting the server at once queue up instead of racing inside llama-cpp
- **VRAM**: `Llama.close()` is called before the model reference is dropped
- **Model-specific defaults** are reachable again — they were being looked up by the friendly `name` instead of the `MODEL_ASSIGNMENTS` key
- **Failed loads** no longer leave `/model/status` reporting a model that isn't there
- **Streaming** dropped its per-token `sleep` and its per-chunk `strip()`; only the final line is trimmed
- **`/model/status` and `/server/info`** report the full effective loading parameters and the live `n_ctx`, including after a lazy load
- **Setup** installs GPU builds from the prebuilt CUDA wheel index (or a real source build) instead of a package that never existed, and verifies GPU offload afterwards
- **`start_server.bat`** works from any working directory and uses the venv's Python directly
- **Logging** is configured once, reaches both handlers, and writes next to `server.py`

### Behaviour changes clients should know about
- Out-of-range inference parameters on `/query` now return `400` instead of being silently discarded
- Unknown model names return `400` (they used to be a `500`) on `/model/load`, `/query`, `/count_tokens`, `/model/parameters` and `/model/inference-parameters`
- Non-streaming generation failures return `500 {"error": ...}` instead of `200` with `"Error: ..."` inside `response`
- `/count_tokens` accepts empty text (was `400`), and its no-model-loaded estimate is budgeted against the model's default `n_ctx` (2048) rather than `max_context_window` (8192)
- `f16_kv` is gone from the API — llama-cpp hasn't accepted it for several versions, so it was validated and then thrown away
- Streaming `partial` values keep their whitespace; only the terminal `response` is stripped

## 🤝 Integration

This server is designed to work with:
- **Android LLM App**: the primary client, with full parameter control
- **Any HTTP client**: plain JSON in, JSON or NDJSON out — `curl`, Python `requests`, a browser fetch
- **Custom applications**: the parameter-discovery endpoints exist so a client can build its own settings UI without hardcoding bounds

## 🛠️ Troubleshooting

### Common Issues

**"Unknown model" (400)**
Use the key from `MODEL_ASSIGNMENTS` (`MyMainLLM`), not the friendly `name` (`kunoichi`). `GET /models` returns the valid list.

**Parameter validation errors (400)**
The message names the bound it broke. Ask the server what it accepts:
```bash
curl http://localhost:5000/model/loading-parameters
curl http://localhost:5000/model/inference-parameters
```

**Model Loading Failures**
- Verify the `model_path` in `config.py` actually exists (`setup_environment.py` checks this for you)
- Check VRAM headroom — lower `n_gpu_layers` or `n_ctx` for large models
- The failure is logged in full; `/model/status` will report `loaded: false` afterwards

**Performance Issues**
- Tune `n_threads` for your CPU
- Raise `n_gpu_layers` until you run out of VRAM, then back off
- Remember only one generation runs at a time — a second request waits for the first

### GPU Issues

If generation is running on the CPU:
1. Ask the installed build directly:
   ```bash
   cd server
   venv\Scripts\python -c "import llama_cpp; print(bool(llama_cpp.llama_supports_gpu_offload()))"
   ```
   `False` means you have a CPU-only wheel — reinstall from the CUDA wheel index (see Quick Start).
2. Confirm the driver is alive with `nvidia-smi`
3. Make sure `n_gpu_layers` isn't set to `0`
4. Watch VRAM in `nvidia-smi` while a model loads — if nothing moves, it's the wheel, not the config

## 📄 License

MIT — see the [LICENSE](LICENSE) file.

## 🔗 Related Projects

- **[Android LLM App](https://github.com/Bloodtailor/my-llm-android-app)** - Mobile client with full parameter control
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** - Core LLM inference library

---

**Note**: This is a personal hobby project built for one home network. It assumes a trusted LAN, a single user, and one model in VRAM at a time. Adequate RAM/VRAM for your chosen GGUF file is the main hardware requirement.
