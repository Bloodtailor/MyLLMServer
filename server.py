from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import logging
import time

# Import our LLM manager
from llm_manager import LLMManager

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        return manager.generate(prompt, system_prompt)
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
            return jsonify({'error': 'No prompt provided'}), 400
        
        logger.info(f"Received prompt: {prompt[:50]}...")
        
        if stream_mode:
            # Stream the response
            return Response(
                stream_with_context(query_llm_stream(prompt, system_prompt, model_name)),
                mimetype='application/x-ndjson'
            )
        else:
            # Non-streaming response
            response = query_llm(prompt, system_prompt, model_name)
            return jsonify({'response': response})
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List all available models."""
    from config import MODEL_ASSIGNMENTS
    return jsonify({'models': list(MODEL_ASSIGNMENTS.keys())})

@app.route('/model/load', methods=['POST'])
def load_model():
    """Load a specific model."""
    try:
        data = request.get_json(force=True)
        model_name = data.get('model', 'MyMainLLM')
        
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
    
    if llm_manager is not None:
        model_name = current_model
        llm_manager = None
        current_model = None
        return jsonify({
            'status': 'success', 
            'message': f'Model {model_name} unloaded successfully'
        })
    else:
        return jsonify({
            'status': 'success', 
            'message': 'No model was loaded'
        })

@app.route('/model/status', methods=['GET'])
def model_status():
    """Get current model status."""
    return jsonify({
        'loaded': llm_manager is not None,
        'current_model': current_model
    })

if __name__ == '__main__':
    # Run the server on all network interfaces so it's accessible from your phone
    logger.info("Starting server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
