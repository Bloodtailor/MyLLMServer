from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import logging
import time
import os
from datetime import datetime

# Import our LLM manager
from llm_manager import LLMManager

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
    
    try:
        logger.info(f"Server info requested by {request.remote_addr}")
        
        # Basic system information
        info = {
            'server_platform': platform.platform(),
            'python_version': platform.python_version(),
            'current_model': current_model,
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
            '/server/info', '/server/ping'
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
        print(f"Starting LLM Server on {ip_address}:5000")
        print(f"Server can be accessed at: http://{ip_address}:5000")
        print(f"Log files are stored in: {os.path.abspath('logs')}")
        print("="*50)
        
        # Run the server on all network interfaces so it's accessible from your phone
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        logger.critical(f"Error starting server: {str(e)}", exc_info=True)