#!/usr/bin/env python
"""
GPU Usage Test - Check actual GPU utilization during model loading and inference
"""

import time
import subprocess
import threading
import requests
import json
from datetime import datetime

class GPUMonitor:
    def __init__(self):
        self.monitoring = False
        self.gpu_usage_data = []
    
    def start_monitoring(self):
        """Start monitoring GPU usage in a separate thread."""
        self.monitoring = True
        self.gpu_usage_data = []
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return collected data."""
        self.monitoring = False
        time.sleep(1)  # Allow final reading
        return self.gpu_usage_data
    
    def _monitor_loop(self):
        """Monitor GPU usage every 0.5 seconds."""
        while self.monitoring:
            try:
                # Get GPU utilization and memory usage
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
                    
                    self.gpu_usage_data.append({
                        'timestamp': timestamp,
                        'gpu_utilization': int(gpu_util),
                        'memory_used_mb': int(mem_used),
                        'memory_total_mb': int(mem_total),
                        'temperature': int(temp)
                    })
                    
            except Exception as e:
                print(f"GPU monitoring error: {e}")
            
            time.sleep(0.5)

def test_direct_llm():
    """Test direct LLM usage with GPU monitoring."""
    print("="*60)
    print("TESTING DIRECT LLM USAGE")
    print("="*60)
    
    monitor = GPUMonitor()
    
    try:
        # Import here to avoid import issues
        from llm_manager import LLMManager
        
        print("Starting GPU monitoring...")
        monitor.start_monitoring()
        
        print("Loading model directly...")
        start_time = time.time()
        
        # Load model with verbose output
        manager = LLMManager.Load("MyMainLLM")
        load_time = time.time() - start_time
        
        print(f"Model loaded in {load_time:.2f} seconds")
        print("Running inference test...")
        
        # Test inference
        inference_start = time.time()
        response = manager.generate("What is artificial intelligence?")
        inference_time = time.time() - inference_start
        
        print(f"Inference completed in {inference_time:.2f} seconds")
        print(f"Response length: {len(response)} characters")
        
        # Stop monitoring
        gpu_data = monitor.stop_monitoring()
        
        return {
            'method': 'direct',
            'load_time': load_time,
            'inference_time': inference_time,
            'response_length': len(response),
            'gpu_usage': gpu_data
        }
        
    except Exception as e:
        monitor.stop_monitoring()
        print(f"Direct test error: {e}")
        return None

def test_flask_llm(server_url="http://localhost:5000"):
    """Test Flask LLM usage with GPU monitoring."""
    print("="*60)
    print("TESTING FLASK LLM USAGE")
    print("="*60)
    
    monitor = GPUMonitor()
    
    try:
        print("Starting GPU monitoring...")
        monitor.start_monitoring()
        
        # Load model via Flask API
        print("Loading model via Flask API...")
        start_time = time.time()
        
        load_response = requests.post(f"{server_url}/model/load", 
                                    json={"model": "MyMainLLM"}, 
                                    timeout=120)
        
        if load_response.status_code != 200:
            print(f"Model load failed: {load_response.status_code}")
            return None
            
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Test inference
        print("Running inference test...")
        inference_start = time.time()
        
        query_response = requests.post(f"{server_url}/query", 
                                     json={
                                         "prompt": "What is artificial intelligence?",
                                         "model": "MyMainLLM",
                                         "stream": False
                                     }, 
                                     timeout=120)
        
        inference_time = time.time() - inference_start
        
        if query_response.status_code == 200:
            response_data = query_response.json()
            response_text = response_data.get('response', '')
            print(f"Inference completed in {inference_time:.2f} seconds")
            print(f"Response length: {len(response_text)} characters")
        else:
            print(f"Query failed: {query_response.status_code}")
            return None
        
        # Stop monitoring
        gpu_data = monitor.stop_monitoring()
        
        return {
            'method': 'flask',
            'load_time': load_time,
            'inference_time': inference_time,
            'response_length': len(response_text),
            'gpu_usage': gpu_data
        }
        
    except Exception as e:
        monitor.stop_monitoring()
        print(f"Flask test error: {e}")
        return None

def analyze_gpu_usage(direct_data, flask_data):
    """Analyze and compare GPU usage between direct and Flask methods."""
    print("="*60)
    print("GPU USAGE ANALYSIS")
    print("="*60)
    
    def analyze_data(data, method_name):
        if not data or not data.get('gpu_usage'):
            print(f"{method_name}: No GPU data available")
            return
            
        gpu_usage = data['gpu_usage']
        if not gpu_usage:
            print(f"{method_name}: No GPU usage recorded")
            return
            
        # Calculate statistics
        gpu_utils = [d['gpu_utilization'] for d in gpu_usage]
        memory_used = [d['memory_used_mb'] for d in gpu_usage]
        
        max_gpu_util = max(gpu_utils) if gpu_utils else 0
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
        max_memory = max(memory_used) if memory_used else 0
        min_memory = min(memory_used) if memory_used else 0
        
        print(f"\n{method_name.upper()} Results:")
        print(f"  Load Time: {data['load_time']:.2f}s")
        print(f"  Inference Time: {data['inference_time']:.2f}s")
        print(f"  Max GPU Utilization: {max_gpu_util}%")
        print(f"  Avg GPU Utilization: {avg_gpu_util:.1f}%")
        print(f"  Memory Range: {min_memory}MB - {max_memory}MB")
        print(f"  Memory Delta: {max_memory - min_memory}MB")
        
        # Show timeline of key events
        print(f"  GPU Usage Timeline:")
        for i, reading in enumerate(gpu_usage[::4]):  # Show every 4th reading
            print(f"    {reading['timestamp']}: {reading['gpu_utilization']}% GPU, {reading['memory_used_mb']}MB")
    
    if direct_data:
        analyze_data(direct_data, "direct")
    if flask_data:
        analyze_data(flask_data, "flask")
    
    # Compare if both available
    if direct_data and flask_data:
        print(f"\nCOMPARISON:")
        speed_ratio = flask_data['inference_time'] / direct_data['inference_time']
        print(f"  Flask is {speed_ratio:.1f}x slower than direct")
        
        direct_max_gpu = max([d['gpu_utilization'] for d in direct_data['gpu_usage']] or [0])
        flask_max_gpu = max([d['gpu_utilization'] for d in flask_data['gpu_usage']] or [0])
        print(f"  Direct max GPU: {direct_max_gpu}%")
        print(f"  Flask max GPU: {flask_max_gpu}%")
        
        direct_memory_delta = max([d['memory_used_mb'] for d in direct_data['gpu_usage']] or [0]) - min([d['memory_used_mb'] for d in direct_data['gpu_usage']] or [0])
        flask_memory_delta = max([d['memory_used_mb'] for d in flask_data['gpu_usage']] or [0]) - min([d['memory_used_mb'] for d in flask_data['gpu_usage']] or [0])
        print(f"  Direct memory increase: {direct_memory_delta}MB")
        print(f"  Flask memory increase: {flask_memory_delta}MB")

def main():
    """Run GPU usage tests."""
    print("GPU Usage Test - Comparing Direct vs Flask LLM Performance")
    print(f"Test started at: {datetime.now()}")
    
    # Test direct usage
    direct_results = test_direct_llm()
    time.sleep(2)  # Brief pause between tests
    
    # Test Flask usage (make sure server is running)
    flask_results = test_flask_llm()
    
    # Analyze results
    analyze_gpu_usage(direct_results, flask_results)
    
    print("\n" + "="*60)
    print("Test completed!")

if __name__ == "__main__":
    main()