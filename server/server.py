from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import logging
import time
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Import our LLM manager
from llm_manager import LLMManager

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up enhanced logging
log_file = os.path.join("logs", f"llm_server_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5),  # 10MB per file, keep 5 backups
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger('llm_server')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
llm_manager = None
current_model = None

def get_llm_manager(model_name="MyMainLLM"):
    """Get or initialize the LLM manager."""
    global llm_manager, current_model
    
    # If we're requesting a different model, unload the current one
    if llm_manager is not None and current_model != model_name:
        logger.info(f"Unloading model {current_model} to load {model_name}")
        llm_manager = None  # Let garbage collection free the memory
    
    if llm_manager is None:
        logger.info(f"Initializing LLM manager with model {model_name}")
        llm_manager = LLMManager.Load(model_name)
        current_model = model_name
        
    return llm_manager

# Function to query the LLM with streaming
def query_llm_stream(prompt, system_prompt="", model_name="MyMainLLM"):
    """Generate a streaming response from the model."""
    try:
        manager = get_llm_manager(model_name)
        
        # Send an immediate status update to prevent timeouts
        yield json.dumps({"status": "processing", "partial": ""}) + "\n"
        
        # Get the streaming generator
        stream = manager.generate_stream(prompt, system_prompt)
        
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
def query_llm(prompt, system_prompt="", model_name="MyMainLLM"):
    """Generate a non-streaming response from the model."""
    try:
        manager = get_llm_manager(model_name)
        response = manager.generate(prompt, system_prompt)
        logger.info(f"Generated non-streaming response of length {len(response)} characters")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/query', methods=['POST'])
def process_query():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        system_prompt = data.get('system_prompt', '')
        model_name = data.get('model', 'MyMainLLM')
        stream_mode = data.get('stream', True)  # Default to streaming
        
        if not prompt:
            logger.warning("Received request with empty prompt")
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Log the first 50 chars of the prompt to avoid huge log files
        logger.info(f"Received prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        if system_prompt:
            logger.info(f"System prompt: {system_prompt[:50]}{'...' if len(system_prompt) > 50 else ''}")
        
        request_ip = request.remote_addr
        logger.info(f"Request from IP: {request_ip}")
        
        if stream_mode:
            # Stream the response
            logger.info(f"Starting streaming response to {request_ip}")
            return Response(
                stream_with_context(query_llm_stream(prompt, system_prompt, model_name)),
                mimetype='application/x-ndjson'
            )
        else:
            # Non-streaming response
            logger.info(f"Starting non-streaming response to {request_ip}")
            response = query_llm(prompt, system_prompt, model_name)
            return jsonify({'response': response})
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models."""
    from config import MODEL_ASSIGNMENTS
    logger.info(f"Model list requested by {request.remote_addr}")
    return jsonify({'models': list(MODEL_ASSIGNMENTS.keys())})

@app.route('/model/load', methods=['POST'])
def load_model():
    """Load a specific model."""
    try:
        data = request.get_json(force=True)
        model_name = data.get('model', 'MyMainLLM')
        
        logger.info(f"Loading model {model_name} requested by {request.remote_addr}")
        
        # This will load the model and unload any previous one
        manager = get_llm_manager(model_name)
        
        return jsonify({
            'status': 'success', 
            'message': f'Model {model_name} loaded successfully',
            'model': model_name
        })
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/unload', methods=['POST'])
def unload_model():
    """Unload the current model."""
    global llm_manager, current_model
    
    logger.info(f"Unloading model requested by {request.remote_addr}")
    
    if llm_manager is not None:
        model_name = current_model
        llm_manager = None
        current_model = None
        logger.info(f"Model {model_name} unloaded successfully")
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
    return jsonify({
        'loaded': llm_manager is not None,
        'current_model': current_model
    })

@app.route('/server/info', methods=['GET'])
def server_info():
    """Get server information."""
    import platform
    import psutil
    
    try:
        logger.info(f"Server info requested by {request.remote_addr}")
        
        # Basic system information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        info = {
            'server_platform': platform.platform(),
            'python_version': platform.python_version(),
            'memory_total': memory.total,
            'memory_available': memory.available,
            'memory_percent': memory.percent,
            'disk_total': disk.total,
            'disk_free': disk.free,
            'disk_percent': disk.percent,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_count': psutil.cpu_count(logical=True),
            'current_model': current_model,
            'model_loaded': llm_manager is not None,
        }
        
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting server info: {str(e)}")
        # If psutil is not installed
        info = {
            'server_platform': platform.platform(),
            'python_version': platform.python_version(),
            'note': 'Install psutil for more detailed system information'
        }
        return jsonify(info)