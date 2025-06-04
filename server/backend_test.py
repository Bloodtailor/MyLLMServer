#!/usr/bin/env python
"""
Backend Test Script - Test the enhanced parameter endpoints using only standard library
"""

import urllib.request
import urllib.parse
import json
import time
from datetime import datetime

def test_endpoint(url, method="GET", data=None, description=""):
    """Test an endpoint and display results using only standard library."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"URL: {method} {url}")
    if data:
        print(f"Data: {json.dumps(data, indent=2)}")
    print(f"{'='*60}")
    
    try:
        if method == "GET":
            # Simple GET request
            with urllib.request.urlopen(url, timeout=10) as response:
                status_code = response.getcode()
                response_data = response.read().decode('utf-8')
                
        elif method == "POST":
            # POST request with JSON data
            if data:
                # Convert data to JSON and encode
                json_data = json.dumps(data).encode('utf-8')
                
                # Create request with proper headers
                req = urllib.request.Request(
                    url,
                    data=json_data,
                    headers={
                        'Content-Type': 'application/json',
                        'Content-Length': len(json_data)
                    },
                    method='POST'
                )
            else:
                # POST request with empty body
                req = urllib.request.Request(url, data=b'', method='POST')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                status_code = response.getcode()
                response_data = response.read().decode('utf-8')
        else:
            print(f"Unsupported method: {method}")
            return False
        
        print(f"Status Code: {status_code}")
        
        if status_code == 200:
            try:
                json_response = json.loads(response_data)
                print("Response:")
                print(json.dumps(json_response, indent=2))
                return True
            except json.JSONDecodeError:
                print("Response (non-JSON):")
                print(response_data)
                return True
        else:
            print(f"Error Response: {response_data}")
            return False
            
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        try:
            error_response = e.read().decode('utf-8')
            print(f"Error Response: {error_response}")
        except:
            pass
        return False
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return False

def main():
    """Run comprehensive backend tests."""
    server_url = "http://localhost:5000"
    
    print("Backend Enhanced Parameter System Test (Standard Library)")
    print(f"Testing server at: {server_url}")
    print(f"Test started at: {datetime.now()}")
    
    # Test 1: Server ping
    success = test_endpoint(
        f"{server_url}/server/ping",
        method="GET",
        description="Server Ping Test"
    )
    
    if not success:
        print("\n❌ Server is not responding. Please start the server first.")
        print("Run: python server.py")
        return
    
    print("\n✅ Server is responding!")
    
    # Test 2: Get loading parameters
    test_endpoint(
        f"{server_url}/model/loading-parameters",
        method="GET",
        description="Get Loading Parameters"
    )
    
    # Test 3: Get inference parameters
    test_endpoint(
        f"{server_url}/model/inference-parameters",
        method="GET",
        description="Get Inference Parameters (no model specified)"
    )
    
    # Test 4: Get inference parameters for specific model
    test_endpoint(
        f"{server_url}/model/inference-parameters?model=MyMainLLM",
        method="GET",
        description="Get Inference Parameters for MyMainLLM"
    )
    
    # Test 5: Load model with custom loading parameters
    load_params = {
        "model": "MyMainLLM",
        "n_ctx": 4096,
        "n_gpu_layers": -1,
        "n_threads": 8,
        "use_mlock": True,
        "use_mmap": True,
        "f16_kv": True
    }
    
    test_endpoint(
        f"{server_url}/model/load",
        method="POST",
        data=load_params,
        description="Load Model with Custom Loading Parameters"
    )
    
    # Test 6: Check model status
    test_endpoint(
        f"{server_url}/model/status",
        method="GET",
        description="Check Model Status After Loading"
    )
    
    # Test 7: Get inference parameters after model is loaded
    test_endpoint(
        f"{server_url}/model/inference-parameters",
        method="GET",
        description="Get Inference Parameters (with loaded model)"
    )
    
    # Test 8: Send query with custom inference parameters
    query_params = {
        "prompt": "What is artificial intelligence?",
        "model": "MyMainLLM",
        "stream": False,
        "temperature": 0.9,
        "max_tokens": 150,
        "top_p": 0.8,
        "top_k": 50,
        "repeat_penalty": 1.2
    }
    
    test_endpoint(
        f"{server_url}/query",
        method="POST",
        data=query_params,
        description="Query with Custom Inference Parameters"
    )
    
    # Test 9: Count tokens
    token_params = {
        "text": "This is a test sentence for counting tokens.",
        "model": "MyMainLLM"
    }
    
    test_endpoint(
        f"{server_url}/count_tokens",
        method="POST",
        data=token_params,
        description="Count Tokens"
    )
    
    # Test 10: Get server info
    test_endpoint(
        f"{server_url}/server/info",
        method="GET",
        description="Get Server Info"
    )
    
    # Test 11: Test parameter validation (should fail gracefully)
    print(f"\n{'='*60}")
    print("Testing Parameter Validation (expect this to fail)")
    print("This test should return an error - that's expected!")
    print(f"{'='*60}")
    
    invalid_params = {
        "model": "MyMainLLM",
        "n_ctx": -500,  # Invalid: negative context
    }
    
    test_endpoint(
        f"{server_url}/model/load",
        method="POST",
        data=invalid_params,
        description="Test Parameter Validation (should fail)"
    )
    
    # Test 12: Test with valid parameters again
    valid_params = {
        "model": "MyMainLLM",
        "n_ctx": 2048,
    }
    
    test_endpoint(
        f"{server_url}/model/load",
        method="POST",
        data=valid_params,
        description="Load Model with Valid Parameters"
    )
    
    # Test 13: Simple query test
    simple_query = {
        "prompt": "Hello, how are you?",
        "model": "MyMainLLM",
        "stream": False,
        "max_tokens": 50
    }
    
    test_endpoint(
        f"{server_url}/query",
        method="POST",
        data=simple_query,
        description="Simple Query Test"
    )
    
    # Test 14: Unload model
    test_endpoint(
        f"{server_url}/model/unload",
        method="POST",
        description="Unload Model"
    )
    
    print(f"\n{'='*60}")
    print("🎉 Backend Testing Complete!")
    print(f"Test finished at: {datetime.now()}")
    print(f"{'='*60}")
    print("\nIf all tests passed, your backend is ready for the Android app!")
    print("If any tests failed, check the server logs for more details.")

if __name__ == "__main__":
    main()