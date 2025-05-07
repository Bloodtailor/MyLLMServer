#!/usr/bin/env python
"""
Benchmark script for testing LLM response generation speeds
"""

import time
import argparse
import statistics
import json
import requests
from typing import List, Dict, Any
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark')

# Import local modules - adjust the path if needed
from llm_manager import LLMManager
from config import MODEL_ASSIGNMENTS, DEFAULT_N_THREADS

class LLMBenchmark:
    def __init__(self, model_name: str = "MyMainLLM", server_url: str = "http://localhost:5000"):
        self.model_name = model_name
        self.server_url = server_url
        self.results = {}
        
    def run_direct_test(self, prompt: str, n_runs: int = 3, stream: bool = False) -> Dict[str, Any]:
        """Test the LLM manager directly."""
        logger.info(f"Testing direct LLM manager with {'streaming' if stream else 'non-streaming'}")
        
        # Load the model
        manager = LLMManager.Load(self.model_name)
        
        times = []
        token_counts = []
        tokens_per_second = []
        
        for i in range(n_runs):
            logger.info(f"Run {i+1}/{n_runs}")
            
            start_time = time.time()
            
            if stream:
                # For streaming, we need to consume the generator
                tokens = 0
                for chunk in manager.generate_stream(prompt):
                    tokens += 1  # Approximation: one token per chunk
                response_length = tokens
            else:
                # For non-streaming
                response = manager.generate(prompt)
                response_length = len(response.split())  # Rough word count approximation
            
            elapsed = time.time() - start_time
            
            # Calculate metrics
            times.append(elapsed)
            token_counts.append(response_length)
            tokens_per_second.append(response_length / elapsed if elapsed > 0 else 0)
            
            logger.info(f"Run {i+1} completed in {elapsed:.2f}s - {response_length} tokens - {response_length/elapsed:.2f} tokens/sec")
        
        # Calculate average metrics
        avg_time = statistics.mean(times)
        avg_tokens = statistics.mean(token_counts)
        avg_tokens_per_sec = statistics.mean(tokens_per_second)
        
        result = {
            "mode": "direct",
            "streaming": stream,
            "runs": n_runs,
            "avg_time_seconds": avg_time,
            "avg_tokens": avg_tokens,
            "avg_tokens_per_second": avg_tokens_per_sec,
            "thread_count": DEFAULT_N_THREADS,
            "all_times": times,
            "all_tokens_per_second": tokens_per_second
        }
        
        logger.info(f"Direct test results: {json.dumps(result, indent=2)}")
        return result
        
    def run_server_test(self, prompt: str, n_runs: int = 3, stream: bool = False) -> Dict[str, Any]:
        """Test the server API."""
        logger.info(f"Testing server API with {'streaming' if stream else 'non-streaming'}")
        
        # Ensure the model is loaded first
        load_url = f"{self.server_url}/model/load"
        load_data = {"model": self.model_name}
        requests.post(load_url, json=load_data)
        
        times = []
        token_counts = []
        tokens_per_second = []
        
        for i in range(n_runs):
            logger.info(f"Run {i+1}/{n_runs}")
            
            # Prepare request data
            request_data = {
                "prompt": prompt,
                "model": self.model_name,
                "stream": stream
            }
            
            start_time = time.time()
            
            if stream:
                # For streaming requests
                response = requests.post(
                    f"{self.server_url}/query", 
                    json=request_data,
                    stream=True
                )
                
                tokens = 0
                for line in response.iter_lines():
                    if line:
                        tokens += 1  # Count chunks as approximation
                response_length = tokens
            else:
                # For non-streaming requests
                response = requests.post(f"{self.server_url}/query", json=request_data)
                response_json = response.json()
                response_text = response_json.get('response', '')
                response_length = len(response_text.split())  # Rough approximation
            
            elapsed = time.time() - start_time
            
            # Calculate metrics
            times.append(elapsed)
            token_counts.append(response_length)
            tokens_per_second.append(response_length / elapsed if elapsed > 0 else 0)
            
            logger.info(f"Run {i+1} completed in {elapsed:.2f}s - {response_length} tokens - {response_length/elapsed:.2f} tokens/sec")
        
        # Calculate average metrics
        avg_time = statistics.mean(times)
        avg_tokens = statistics.mean(token_counts)
        avg_tokens_per_sec = statistics.mean(tokens_per_second)
        
        result = {
            "mode": "server",
            "streaming": stream,
            "runs": n_runs,
            "avg_time_seconds": avg_time,
            "avg_tokens": avg_tokens,
            "avg_tokens_per_second": avg_tokens_per_sec,
            "all_times": times,
            "all_tokens_per_second": tokens_per_second
        }
        
        logger.info(f"Server test results: {json.dumps(result, indent=2)}")
        return result
    
    def run_all_tests(self, prompt: str, n_runs: int = 3) -> Dict[str, Any]:
        """Run all benchmark tests."""
        logger.info(f"Starting comprehensive benchmark with {n_runs} runs each")
        
        # Run all test configurations
        results = {
            "direct_non_streaming": self.run_direct_test(prompt, n_runs, stream=False),
            "direct_streaming": self.run_direct_test(prompt, n_runs, stream=True),
            "server_non_streaming": self.run_server_test(prompt, n_runs, stream=False),
            "server_streaming": self.run_server_test(prompt, n_runs, stream=True)
        }
        
        # Compute comparisons
        comparisons = self._compute_comparisons(results)
        
        # Combine results
        benchmark_results = {
            "test_info": {
                "model": self.model_name,
                "prompt": prompt,
                "runs_per_test": n_runs,
                "thread_count": DEFAULT_N_THREADS,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": results,
            "comparisons": comparisons
        }
        
        # Save results to file
        self._save_results(benchmark_results)
        
        return benchmark_results
    
    def _compute_comparisons(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comparison metrics between different test configurations."""
        comparisons = {}
        
        # Streaming vs Non-streaming for direct mode
        direct_speedup = results["direct_non_streaming"]["avg_tokens_per_second"] / \
                        results["direct_streaming"]["avg_tokens_per_second"] \
                        if results["direct_streaming"]["avg_tokens_per_second"] > 0 else 0
                        
        comparisons["direct_streaming_vs_non_streaming"] = {
            "speedup_factor": direct_speedup,
            "conclusion": "Non-streaming is faster" if direct_speedup > 1 else "Streaming is faster"
        }
        
        # Streaming vs Non-streaming for server mode
        server_speedup = results["server_non_streaming"]["avg_tokens_per_second"] / \
                        results["server_streaming"]["avg_tokens_per_second"] \
                        if results["server_streaming"]["avg_tokens_per_second"] > 0 else 0
                        
        comparisons["server_streaming_vs_non_streaming"] = {
            "speedup_factor": server_speedup,
            "conclusion": "Non-streaming is faster" if server_speedup > 1 else "Streaming is faster"
        }
        
        # Direct vs Server for non-streaming
        non_streaming_speedup = results["direct_non_streaming"]["avg_tokens_per_second"] / \
                              results["server_non_streaming"]["avg_tokens_per_second"] \
                              if results["server_non_streaming"]["avg_tokens_per_second"] > 0 else 0
                              
        comparisons["direct_vs_server_non_streaming"] = {
            "speedup_factor": non_streaming_speedup,
            "conclusion": "Direct is faster" if non_streaming_speedup > 1 else "Server is faster"
        }
        
        # Direct vs Server for streaming
        streaming_speedup = results["direct_streaming"]["avg_tokens_per_second"] / \
                          results["server_streaming"]["avg_tokens_per_second"] \
                          if results["server_streaming"]["avg_tokens_per_second"] > 0 else 0
                          
        comparisons["direct_vs_server_streaming"] = {
            "speedup_factor": streaming_speedup,
            "conclusion": "Direct is faster" if streaming_speedup > 1 else "Server is faster"
        }
        
        return comparisons
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to a file."""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"benchmark_{self.model_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filename}")

def main():
    """Main function to run benchmarks from command line."""
    parser = argparse.ArgumentParser(description="Benchmark LLM performance")
    parser.add_argument("--model", type=str, default="MyMainLLM", help="Model name to benchmark")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms", help="Prompt to use")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test")
    parser.add_argument("--server", type=str, default="http://localhost:5000", help="Server URL")
    parser.add_argument("--mode", type=str, choices=["all", "direct", "server"], default="all", help="Test mode")
    parser.add_argument("--stream", type=bool, default=None, help="Test streaming (if not all)")
    
    args = parser.parse_args()
    
    benchmark = LLMBenchmark(model_name=args.model, server_url=args.server)
    
    if args.mode == "all":
        results = benchmark.run_all_tests(args.prompt, args.runs)
        # Print summary
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        for test_name, comparison in results["comparisons"].items():
            print(f"{test_name}: {comparison['conclusion']} (factor: {comparison['speedup_factor']:.2f}x)")
        
        print("\nDetailed results:")
        for test_name, result in results["results"].items():
            print(f"  {test_name}: {result['avg_tokens_per_second']:.2f} tokens/sec")
        
    elif args.mode == "direct":
        stream = args.stream if args.stream is not None else False
        benchmark.run_direct_test(args.prompt, args.runs, stream)
    elif args.mode == "server":
        stream = args.stream if args.stream is not None else False
        benchmark.run_server_test(args.prompt, args.runs, stream)

if __name__ == "__main__":
    main()