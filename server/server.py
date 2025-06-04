from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import logging
import time
import os
import platform
import subprocess
import inspect
from datetime import datetime

# Import our LLM manager and config
from llm_manager import LLMManager
from config import (
    MODEL_ASSIGNMENTS,
    GLOBAL_LOADING_PARAMETERS,
    GLOBAL_INFERENCE_PARAMETERS,
    get_loading_parameter_defaults,
    get_inference_parameter_defaults,
    validate_parameter
)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up enhanced logging
log_file = os.path.join("logs", f"llm_server_{datetime.now().strftime('%Y%m%d')}.log")

# Configure basic logging first to catch any early errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Create logger
logger = logging.getLogger('llm_server')
logger.setLevel(logging.INFO)

# Create handlers
try:
    # Try to create a rotating file handler
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10485760,  # 10MB per file
        backupCount=5       # Keep 5 backups
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Logging setup complete with file and console handlers")
except Exception as e:
    logger.warning(f"Could not set up rotating file handler: {str(e)}")
    logger.info("Continuing with console logging only")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
llm_manager = None
current_model = None
current_loading_params = None

def get_llm_manager(model_name="MyMainLLM", loading_params=None):
    """Get or initialize the LLM manager."""
    global llm_manager, current_model, current_loading_params
    
    # If we're requesting a different model or loading parameters, unload the current one
    if (llm_manager is not None and 
        (current_model != model_name or current_loading_params != loading_params)):
        logger.info(f"Unloading model {current_model} to load {model_name} with new parameters")
        llm_manager = None  # Let garbage collection free the memory
    
    if llm_manager is None:
        logger.info(f"Initializing LLM manager with model {model_name} and loading parameters {loading_params}")
        llm_manager = LLMManager.Load(model_name, loading_params)
        current_model = model_name
        current_loading_params = loading_params
        
    return llm_manager

# Function to query the LLM with streaming
def query_llm_stream(prompt, system_prompt="", model_name="MyMainLLM", inference_params=None):
    """Generate a streaming response from the model using raw prompt."""
    try:
        manager = get_llm_manager(model_name, current_loading_params)
        
        # Send an immediate status update to prevent timeouts
        yield json.dumps({"status": "processing", "partial": ""}) + "\n"
        
        # Prepare inference parameters
        kwargs = inference_params or {}
        
        # Send the raw prompt directly to the model without formatting
        # The model will receive exactly what the user typed
        stream = manager.generate_stream_raw(prompt, system_prompt, **kwargs)
        
        # Stream the response chunks
        partial_response = ""
        for chunk in stream:
            if "error" in chunk:
                yield json.dumps({"status": "error", "error": chunk["error"]}) + "\n"
                return
                
            partial_response = chunk["text"]
            yield json.dumps({"status": "generating", "partial": partial_response}) + "\n"
            
            # Small delay to avoid overwhelming the client
            time.sleep(0.01)
            
        # Final complete response
        logger.info(f"Generated response of length {len(partial_response)} characters")
        yield json.dumps({"status": "complete", "response": partial_response}) + "\n"
        
    except Exception as e:
        error_msg = f"Error generating streaming response: {str(e)}"
        logger.error(error_msg)
        yield json.dumps({"status": "error", "error": error_msg}) + "\n"

# Function to query the LLM (non-streaming)
def query_llm(prompt, system_prompt="", model_name="MyMainLLM", inference_params=None):
    """Generate a non-streaming response from the model using raw prompt."""
    try:
        manager = get_llm_manager(model_name, current_loading_params)
        
        # Prepare inference parameters
        kwargs = inference_params or {}
        
        # Send the raw prompt directly without formatting
        response = manager.generate_raw(prompt, system_prompt, **kwargs)
        logger.info(f"Generated non-streaming response of length {len(response)} characters")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}"

# ============================================================================
# NEW PARAMETER ENDPOINTS
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
        model_name = request.args.get('model', current_model or 'MyMainLLM')
        logger.info(f"Inference parameters requested for {model_name} by {request.remote_addr}")
        
        # Get current values if model is loaded, otherwise use defaults
        if llm_manager is not None and current_model == model_name:
            current_defaults = llm_manager.get_inference_parameter_defaults()
        else:
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
        
    except Exception as e:
        logger.error(f"Error getting inference parameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# UPDATED ENDPOINTS
# ============================================================================

@app.route('/query', methods=['POST'])
def process_query():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        system_prompt = data.get('system_prompt', '')
        model_name = data.get('model', current_model or 'MyMainLLM')
        stream_mode = data.get('stream', True)  # Default to streaming
        
        # Extract inference parameters from request
        inference_params = {}
        for param_name in GLOBAL_INFERENCE_PARAMETERS.keys():
            if param_name in data:
                try:
                    # Validate the parameter
                    validated_value = validate_parameter(
                        param_name, data[param_name], GLOBAL_INFERENCE_PARAMETERS[param_name]
                    )
                    inference_params[param_name] = validated_value
                except ValueError as e:
                    logger.warning(f"Invalid inference parameter {param_name}: {str(e)}")
                    # Continue without this parameter
        
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
        model_name = data.get('model', 'MyMainLLM')
        
        # Extract loading parameters from request
        loading_params = {}
        for param_name in GLOBAL_LOADING_PARAMETERS.keys():
            if param_name in data:
                loading_params[param_name] = data[param_name]
        
        # Also check for model-specific parameters
        if model_name in MODEL_ASSIGNMENTS and "loading_params" in MODEL_ASSIGNMENTS[model_name]:
            for param_name in MODEL_ASSIGNMENTS[model_name]["loading_params"].keys():
                if param_name in data:
                    loading_params[param_name] = data[param_name]
        
        logger.info(f"Loading model {model_name} with parameters {loading_params} requested by {request.remote_addr}")
        
        # This will load the model and unload any previous one
        manager = get_llm_manager(model_name, loading_params)
        
        # Get the actual loading parameters used
        actual_params = manager.get_loading_parameters()
        
        return jsonify({
            'status': 'success', 
            'message': f'Model {model_name} loaded successfully',
            'model': model_name,
            'loading_parameters': actual_params
        })
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/unload', methods=['POST'])
def unload_model():
    """Unload the current model."""
    global llm_manager, current_model, current_loading_params
    
    logger.info(f"Unloading model requested by {request.remote_addr}")
    
    if llm_manager is not None:
        model_name = current_model
        loading_params = current_loading_params
        llm_manager = None
        current_model = None
        current_loading_params = None
        logger.info(f"Model {model_name} with loading parameters {loading_params} unloaded successfully")
        return jsonify({
            'status': 'success', 
            'message': f'Model {model_name} unloaded successfully'
        })
    else:
        logger.info("No model was loaded to unload")
        return jsonify({
            'status': 'success', 
            'message': 'No model was loaded'
        })

@app.route('/model/status', methods=['GET'])
def model_status():
    """Get current model status."""
    logger.info(f"Model status requested by {request.remote_addr}")
    
    # Get current context length from loading parameters
    context_length = None
    if current_loading_params and "n_ctx" in current_loading_params:
        context_length = current_loading_params["n_ctx"]
    
    return jsonify({
        'loaded': llm_manager is not None,
        'current_model': current_model,
        'context_length': context_length,
        'loading_parameters': current_loading_params
    })

@app.route('/count_tokens', methods=['POST'])
def count_tokens():
    """Count tokens in a text string and return context usage."""
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        model_name = data.get('model', current_model or 'MyMainLLM')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get token count - if model is loaded, use it; otherwise estimate
        if llm_manager is not None and current_model == model_name:
            manager = llm_manager
            token_count = manager.count_tokens(text)
            context_usage = manager.get_context_usage(text)
        else:
            # Create a temporary manager without loading the model for estimation
            from config import MODEL_ASSIGNMENTS
            from llm_manager import ModelConfig, LLMManager
            config = ModelConfig(**MODEL_ASSIGNMENTS[model_name])
            manager = LLMManager(config)
            token_count = manager.count_tokens(text)  # This will use rough estimation
            
            # Calculate usage statistics manually
            max_context = config.max_context_window
            usage_percentage = (token_count / max_context) * 100 if max_context > 0 else 0
            
            context_usage = {
                'token_count': token_count,
                'max_context': max_context,
                'usage_percentage': round(usage_percentage, 1),
                'remaining_tokens': max_context - token_count
            }
        
        return jsonify({
            'text': text,
            'model': model_name,
            'context_usage': context_usage
        })
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/server/info', methods=['GET'])
def server_info():
    """Get server information."""
    import platform
    
    try:
        logger.info(f"Server info requested by {request.remote_addr}")
        
        # Basic system information
        info = {
            'server_platform': platform.platform(),
            'python_version': platform.python_version(),
            'current_model': current_model,
            'loading_parameters': current_loading_params,
            'model_loaded': llm_manager is not None,
        }
        
        # Try to get additional system info if psutil is available
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Add psutil information
            info.update({
                'memory_total': memory.total,
                'memory_available': memory.available,
                'memory_percent': memory.percent,
                'disk_total': disk.total,
                'disk_free': disk.free,
                'disk_percent': disk.percent,
                'cpu_percent': psutil.cpu_percent(interval=1),
                'cpu_count': psutil.cpu_count(logical=True),
            })
        except ImportError:
            info['note'] = 'Install psutil for more detailed system information'
        
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting server info: {str(e)}")
        return jsonify({
            'error': str(e),
            'server_platform': platform.platform(),
            'python_version': platform.python_version(),
        })

@app.route('/server/ping', methods=['GET'])
def ping():
    """Simple ping endpoint to check server status."""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat()
    })
    
@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors gracefully."""
    logger.warning(f"404 error: {request.path} not found")
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/query', '/models', '/model/load', 
            '/model/unload', '/model/status',
            '/count_tokens', '/server/info', '/server/ping',
            '/model/loading-parameters', '/model/inference-parameters'
        ]
    }), 404
    
@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors gracefully."""
    logger.error(f"500 error: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500

@app.route('/model/parameters', methods=['GET'])
def get_model_parameters():
    """Get model prefix/suffix parameters for the current or specified model."""
    try:
        model_name = request.args.get('model', current_model or 'MyMainLLM')
        
        if model_name not in MODEL_ASSIGNMENTS:
            return jsonify({'error': f'Unknown model: {model_name}'}), 400
        
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
        
    except Exception as e:
        logger.error(f"Error getting model parameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
        print(f"Starting Enhanced LLM Server on {ip_address}:5000")
        print(f"Server can be accessed at: http://{ip_address}:5000")
        print(f"Log files are stored in: {os.path.abspath('logs')}")
        print("\nNew endpoints available:")
        print(f"  GET  /model/loading-parameters")
        print(f"  GET  /model/inference-parameters")
        print(f"  POST /model/load (enhanced with loading parameters)")
        print(f"  POST /query (enhanced with inference parameters)")
        print("="*50)
        
        # Run the server on all network interfaces so it's accessible from your phone
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        logger.critical(f"Error starting server: {str(e)}", exc_info=True)