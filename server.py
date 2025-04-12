from flask import Flask, request, jsonify
from flask_cors import CORS
import json

# Import your LLM query function here
# from your_llm_module import query_llm

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Placeholder LLM query function (replace with your actual function)
def query_llm(prompt):
    # Replace this with your actual LLM query code
    return f"Response to: {prompt}\n\nThis is a placeholder response. Replace with your actual LLM integration."

@app.route('/query', methods=['POST'])
def process_query():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Query your LLM
        response = query_llm(prompt)
        
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the server on all network interfaces so it's accessible from your phone
    # Change the port if needed
    app.run(host='0.0.0.0', port=5000, debug=True)
