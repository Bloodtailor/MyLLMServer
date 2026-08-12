from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import json
import logging
import os
import platform
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Import our LLM manager and config
from llm_manager import LLMManager
from config import (
    MODEL_ASSIGNMENTS,
    GLOBAL_LOADING_PARAMETERS,
    GLOBAL_INFERENCE_PARAMETERS,
    ParameterValidationError,
    SERVER_HOST,
    SERVER_PORT,
    get_default_n_ctx,
    get_inference_parameter_defaults,
    get_model_loading_parameter_definitions,
    validate_parameter
)

# Paths are anchored to this file so the server works from any working directory
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SERVER_DIR, "logs")

def setup_logging():
    """Configure the root logger once: rotating file + a single console handler."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        return

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"llm_server_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB per file
            backupCount=5       # Keep 5 backups
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception as e:
        root.warning(f"Could not set up rotating file handler: {str(e)} - continuing with console logging only")

def prime_cpu_percent():
    """Take and discard one cpu_percent sample.

    psutil.cpu_percent(interval=None) is comparative and returns a flat 0.0 on
    its first call in a process, so /server/info would report a false 0.0 on the
    first hit after startup without a baseline.
    """
    try:
        import psutil
        psutil.cpu_percent(interval=None)
    except Exception:
        pass  # psutil is optional; /server/info degrades on its own

setup_logging()
prime_cpu_percent()
logger = logging.getLogger('llm_server')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model state. Everything that touches the Llama instance (load, unload,
# generation) is serialized by model_lock: llama-cpp is not thread-safe and Flask
# serves requests on threads. Reentrant purely as a safety net - no lock-holding
# path re-acquires it today (the helpers below all document "caller must hold").
model_lock = threading.RLock()
llm_manager = None
current_model = None

# Derived from config rather than duplicated, so renaming a key in
# MODEL_ASSIGNMENTS can't leave a dangling default that only shows up later as
# per-request 400s.
if not MODEL_ASSIGNMENTS:
    raise SystemExit("MODEL_ASSIGNMENTS is empty (check config.py)")
DEFAULT_MODEL = next(iter(MODEL_ASSIGNMENTS))

class UnknownModelError(ValueError):
    """Raised when a request names a model that isn't in MODEL_ASSIGNMENTS."""

def resolve_model_name(requested):
    """Resolve the model a request refers to, raising UnknownModelError if it's bogus."""
    model_name = requested or current_model or DEFAULT_MODEL
    if model_name not in MODEL_ASSIGNMENTS:
        raise UnknownModelError(f"Unknown model: {model_name}")
    return model_name

def extract_parameters(data, definitions):
    """Pull and validate known parameters out of a request body.

    Returns (validated params, list of error messages for the invalid ones).
    """
    params = {}
    errors = []
    for param_name, param_def in definitions.items():
        if param_name in data:
            try:
                params[param_name] = validate_parameter(param_name, data[param_name], param_def)
            except ValueError as e:
                errors.append(str(e))
    return params, errors

def unload_current_model():
    """Release the loaded model. Caller must hold model_lock."""
    global llm_manager, current_model

    if llm_manager is not None:
        llm_manager.close()
    llm_manager = None
    current_model = None

def needs_reload(model_name, loading_params):
    """Decide whether the requested model/parameters differ from what's loaded."""
    if llm_manager is None or current_model != model_name:
        return True
    if not loading_params:
        return False
    effective = llm_manager.get_loading_parameters()
    return any(effective.get(name) != value for name, value in loading_params.items())

def get_llm_manager(model_name=DEFAULT_MODEL, loading_params=None):
    """Get or initialize the LLM manager. Caller must hold model_lock."""
    global llm_manager, current_model

    if model_name not in MODEL_ASSIGNMENTS:
        raise UnknownModelError(f"Unknown model: {model_name}")

    if not needs_reload(model_name, loading_params):
        return llm_manager

    if llm_manager is not None:
        logger.info(f"Unloading model {current_model} to load {model_name} with new parameters")
        unload_current_model()

    logger.info(f"Initializing LLM manager with model {model_name} and loading parameters {loading_params}")
    try:
        llm_manager = LLMManager.Load(model_name, loading_params)
        current_model = model_name
    except Exception:
        # Never leave the globals pointing at a model that isn't there
        llm_manager = None
        current_model = None
        raise

    return llm_manager

# Function to query the LLM with streaming
def query_llm_stream(prompt, system_prompt="", model_name=DEFAULT_MODEL, inference_params=None):
    """Generate a streaming response from the model using raw prompt."""
    # Sent before acquiring the lock or loading so the client gets bytes immediately
    yield json.dumps({"status": "processing", "partial": ""}) + "\n"

    # Held for the whole generation - a concurrent load/unload waits for us
    model_lock.acquire()
    try:
        manager = get_llm_manager(model_name)

        kwargs = inference_params or {}

        # Send the raw prompt directly to the model without formatting
        # The model will receive exactly what the user typed
        stream = manager.generate_stream_raw(prompt, system_prompt, **kwargs)

        partial_response = ""
        for chunk in stream:
            if "error" in chunk:
                yield json.dumps({"status": "error", "error": chunk["error"]}) + "\n"
                return

            partial_response = chunk["text"]
            yield json.dumps({"status": "generating", "partial": partial_response}) + "\n"

        final_response = partial_response.strip()
        logger.info(f"Generated response of length {len(final_response)} characters")
        yield json.dumps({"status": "complete", "response": final_response}) + "\n"

    except Exception as e:
        error_msg = f"Error generating streaming response: {str(e)}"
        logger.error(error_msg)
        yield json.dumps({"status": "error", "error": error_msg}) + "\n"
    finally:
        model_lock.release()

# Function to query the LLM (non-streaming)
def query_llm(prompt, system_prompt="", model_name=DEFAULT_MODEL, inference_params=None):
    """Generate a non-streaming response from the model using raw prompt."""
    with model_lock:
        manager = get_llm_manager(model_name)

        kwargs = inference_params or {}

        # Send the raw prompt directly without formatting
        response = manager.generate_raw(prompt, system_prompt, **kwargs)
        logger.info(f"Generated non-streaming response of length {len(response)} characters")
        return response

# ============================================================================
# PARAMETER ENDPOINTS
# ============================================================================

@app.route('/model/loading-parameters', methods=['GET'])
def get_loading_parameters():
    """Get all available loading parameters with their defaults and limits."""
    try:
        logger.info(f"Loading parameters requested by {request.remote_addr}")

        # Build response with global defaults and model-specific parameters
        response = {
            "global_defaults": GLOBAL_LOADING_PARAMETERS.copy(),
            "model_specific": {}
        }

        # Add model-specific loading parameters
        for model_name, model_config in MODEL_ASSIGNMENTS.items():
            if "loading_params" in model_config:
                response["model_specific"][model_name] = model_config["loading_params"]

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error getting loading parameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/inference-parameters', methods=['GET'])
def get_inference_parameters():
    """Get all inference parameters for current or specified model."""
    try:
        model_name = resolve_model_name(request.args.get('model'))
        logger.info(f"Inference parameters requested for {model_name} by {request.remote_addr}")

        current_defaults = get_inference_parameter_defaults(model_name)

        # Build response with parameter definitions and current values
        parameters = {}
        for param_name, param_def in GLOBAL_INFERENCE_PARAMETERS.items():
            current_value = current_defaults.get(param_name, param_def["default"])
            parameters[param_name] = {
                "current": current_value,
                "default": param_def["default"],
                "min": param_def.get("min"),
                "max": param_def.get("max"),
                "type": param_def.get("type", "float"),
                "description": param_def.get("description", "")
            }

        response = {
            "model": model_name,
            "parameters": parameters
        }

        return jsonify(response)

    except UnknownModelError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error getting inference parameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MODEL / QUERY ENDPOINTS
# ============================================================================

@app.route('/query', methods=['POST'])
def process_query():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        system_prompt = data.get('system_prompt', '')
        model_name = resolve_model_name(data.get('model'))
        stream_mode = data.get('stream', True)  # Default to streaming

        # Extract inference parameters from request
        inference_params, param_errors = extract_parameters(data, GLOBAL_INFERENCE_PARAMETERS)
        if param_errors:
            logger.warning(f"Invalid inference parameters: {param_errors}")
            return jsonify({
                'error': 'Invalid inference parameters: ' + '; '.join(param_errors),
                'invalid_parameters': param_errors
            }), 400

        if not prompt:
            logger.warning("Received request with empty prompt")
            return jsonify({'error': 'No prompt provided'}), 400

        # Log the first 50 chars of the prompt to avoid huge log files
        logger.info(f"Received raw prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        if system_prompt:
            logger.info(f"System prompt: {system_prompt[:50]}{'...' if len(system_prompt) > 50 else ''}")
        if inference_params:
            logger.info(f"Custom inference parameters: {inference_params}")

        request_ip = request.remote_addr
        logger.info(f"Request from IP: {request_ip}")

        if stream_mode:
            # Stream the response
            logger.info(f"Starting streaming response to {request_ip}")
            return Response(
                stream_with_context(query_llm_stream(prompt, system_prompt, model_name, inference_params)),
                mimetype='application/x-ndjson'
            )
        else:
            # Non-streaming response
            logger.info(f"Starting non-streaming response to {request_ip}")
            response = query_llm(prompt, system_prompt, model_name, inference_params)
            return jsonify({'response': response})

    except UnknownModelError as e:
        return jsonify({'error': str(e)}), 400
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models."""
    logger.info(f"Model list requested by {request.remote_addr}")
    return jsonify({'models': list(MODEL_ASSIGNMENTS.keys())})

@app.route('/model/load', methods=['POST'])
def load_model():
    """Load a specific model with custom loading parameters."""
    try:
        data = request.get_json(force=True)
        model_name = resolve_model_name(data.get('model') or DEFAULT_MODEL)

        # Extract loading parameters from request (global + this model's own)
        loading_params, param_errors = extract_parameters(
            data, get_model_loading_parameter_definitions(model_name)
        )
        if param_errors:
            logger.warning(f"Invalid loading parameters: {param_errors}")
            return jsonify({
                'error': 'Invalid loading parameters: ' + '; '.join(param_errors),
                'invalid_parameters': param_errors
            }), 400

        logger.info(f"Loading model {model_name} with parameters {loading_params} requested by {request.remote_addr}")

        # This will load the model and unload any previous one
        with model_lock:
            manager = get_llm_manager(model_name, loading_params)
            actual_params = manager.get_loading_parameters()

        return jsonify({
            'status': 'success',
            'message': f'Model {model_name} loaded successfully',
            'model': model_name,
            'loading_parameters': actual_params
        })
    except UnknownModelError as e:
        return jsonify({'error': str(e)}), 400
    except ParameterValidationError as e:
        # Only the validation path is a client error. A bare ValueError here is
        # llama-cpp's own (e.g. "Model path does not exist") and is a 500.
        logger.warning(f"Invalid loading parameters: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/unload', methods=['POST'])
def unload_model():
    """Unload the current model."""
    logger.info(f"Unloading model requested by {request.remote_addr}")

    # Blocks until any in-flight generation finishes
    with model_lock:
        if llm_manager is None:
            logger.info("No model was loaded to unload")
            return jsonify({
                'status': 'success',
                'message': 'No model was loaded'
            })

        model_name = current_model
        loading_params = llm_manager.get_loading_parameters()
        unload_current_model()

    logger.info(f"Model {model_name} with loading parameters {loading_params} unloaded successfully")
    return jsonify({
        'status': 'success',
        'message': f'Model {model_name} unloaded successfully'
    })

@app.route('/model/status', methods=['GET'])
def model_status():
    """Get current model status."""
    try:
        logger.info(f"Model status requested by {request.remote_addr}")

        manager = llm_manager
        loading_parameters = manager.get_loading_parameters() if manager is not None else None
        context_length = manager.get_max_context() if manager is not None else None

        return jsonify({
            'loaded': manager is not None,
            'current_model': current_model,
            'context_length': context_length,
            'loading_parameters': loading_parameters
        })
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/count_tokens', methods=['POST'])
def count_tokens():
    """Count tokens in a text string and return context usage."""
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        if not isinstance(text, str):
            return jsonify({'error': 'text must be a string'}), 400

        model_name = resolve_model_name(data.get('model'))

        # Tokenizing only reads the model's vocab, so this deliberately does not
        # take model_lock - the app calls it on every keystroke. The manager's
        # own instance lock still keeps it from racing a concurrent unload/free.
        manager = llm_manager
        if manager is not None and current_model == model_name:
            context_usage = manager.get_context_usage(text)
        else:
            # No model loaded: estimate tokens and budget against the n_ctx this
            # model would actually be loaded with (not its max_context_window).
            token_count = len(text) // 3
            max_context = get_default_n_ctx(model_name)
            usage_percentage = (token_count / max_context) * 100 if max_context > 0 else 0

            context_usage = {
                'token_count': token_count,
                'max_context': max_context,
                'usage_percentage': round(usage_percentage, 1),
                'remaining_tokens': max(0, max_context - token_count)
            }

        return jsonify({
            'text': text,
            'model': model_name,
            'context_usage': context_usage
        })
    except UnknownModelError as e:
        return jsonify({'error': str(e)}), 400
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/server/info', methods=['GET'])
def server_info():
    """Get server information."""
    try:
        logger.info(f"Server info requested by {request.remote_addr}")

        manager = llm_manager

        # Basic system information
        info = {
            'server_platform': platform.platform(),
            'python_version': platform.python_version(),
            'current_model': current_model,
            'loading_parameters': manager.get_loading_parameters() if manager is not None else None,
            'model_loaded': manager is not None,
        }

        # Try to get additional system info if psutil is available
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(SERVER_DIR)

            # Add psutil information
            info.update({
                'memory_total': memory.total,
                'memory_available': memory.available,
                'memory_percent': memory.percent,
                'disk_total': disk.total,
                'disk_free': disk.free,
                'disk_percent': disk.percent,
                'cpu_percent': psutil.cpu_percent(interval=None),
                'cpu_count': psutil.cpu_count(logical=True),
            })
        except ImportError:
            info['note'] = 'Install psutil for more detailed system information'
        except Exception as e:
            # A psutil hiccup (e.g. a drive that went away) degrades to the basic
            # response rather than collapsing the whole endpoint.
            logger.warning(f"Could not collect system details: {str(e)}")
            info['note'] = f'System details unavailable: {str(e)}'

        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting server info: {str(e)}")
        return jsonify({
            'error': str(e),
            'server_platform': platform.platform(),
            'python_version': platform.python_version(),
        }), 500

@app.route('/server/ping', methods=['GET'])
def ping():
    """Simple ping endpoint to check server status."""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model/parameters', methods=['GET'])
def get_model_parameters():
    """Get model prefix/suffix parameters for the current or specified model."""
    try:
        model_name = resolve_model_name(request.args.get('model'))

        model_config = MODEL_ASSIGNMENTS[model_name]
        inference_params = model_config.get('inference_params', {})

        # Extract prefix/suffix parameters
        parameters = {
            'model': model_name,
            'pre_prompt_prefix': inference_params.get('pre_prompt_prefix', ''),
            'pre_prompt_suffix': inference_params.get('pre_prompt_suffix', ''),
            'input_prefix': inference_params.get('input_prefix', ''),
            'input_suffix': inference_params.get('input_suffix', ''),
            'assistant_prefix': inference_params.get('assistant_prefix', ''),
            'assistant_suffix': inference_params.get('assistant_suffix', '')
        }

        logger.info(f"Model parameters requested for {model_name} by {request.remote_addr}")
        return jsonify(parameters)

    except UnknownModelError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error getting model parameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors gracefully."""
    logger.warning(f"404 error: {request.path} not found")
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': sorted(
            str(rule) for rule in app.url_map.iter_rules() if rule.endpoint != 'static'
        )
    }), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors gracefully."""
    logger.error(f"500 error: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500

def get_ip_address():
    """Get the server's IP address to display in the console."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == '__main__':
    # Print server information
    try:
        ip_address = get_ip_address()
        print("="*50)
        print(f"Starting Enhanced LLM Server on {ip_address}:{SERVER_PORT}")
        print(f"Server can be accessed at: http://{ip_address}:{SERVER_PORT}")
        print(f"Log files are stored in: {LOG_DIR}")
        print("\nEndpoints available:")
        for rule in sorted(str(r) for r in app.url_map.iter_rules() if r.endpoint != 'static'):
            print(f"  {rule}")
        print("="*50)

        # Bound to all interfaces so the phone can reach it, so debug must stay off:
        # the Werkzeug debugger would be a remote shell for the whole LAN.
        app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        logger.critical(f"Error starting server: {str(e)}", exc_info=True)
