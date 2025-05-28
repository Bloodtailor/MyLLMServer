#!/usr/bin/env python
"""
GPU Environment Diagnostic Tool
Compares GPU environment between direct execution and Flask execution
"""

import os
import sys
import platform
import subprocess
import logging
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_cuda_environment():
    """Check CUDA environment variables and paths."""
    print_section("CUDA Environment Check")
    
    cuda_vars = [
        'CUDA_PATH', 'CUDA_HOME', 'CUDA_ROOT',
        'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
        'PATH', 'LD_LIBRARY_PATH'
    ]
    
    for var in cuda_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"{var}: {value}")
    
    # Check if nvidia-smi is available
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"\nnvidia-smi: AVAILABLE")
            # Get GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA GeForce' in line or 'GPU' in line:
                    print(f"GPU Info: {line.strip()}")
        else:
            print(f"\nnvidia-smi: FAILED (return code: {result.returncode})")
    except Exception as e:
        print(f"\nnvidia-smi: ERROR - {str(e)}")

def check_python_environment():
    """Check Python environment details."""
    print_section("Python Environment Check")
    
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Process ID: {os.getpid()}")
    print(f"Working Directory: {os.getcwd()}")

def check_llama_cpp_installation():
    """Check llama-cpp-python installation and CUDA support."""
    print_section("llama-cpp-python Installation Check")
    
    try:
        import llama_cpp
        print(f"llama-cpp-python: IMPORTED SUCCESSFULLY")
        print(f"Version: {getattr(llama_cpp, '__version__', 'Unknown')}")
        print(f"Module Path: {llama_cpp.__file__}")
        
        # Try to check for CUDA support
        try:
            # Create a simple Llama instance to test GPU availability
            print("\nTesting GPU availability...")
            test_model_path = "C:/Users/soulo/.cache/lm-studio/models/TheBloke/Kunoichi-7B-GGUF/kunoichi-7b.Q6_K.gguf"
            
            if os.path.exists(test_model_path):
                print(f"Test model exists: {test_model_path}")
                
                # Test with GPU layers
                try:
                    from llama_cpp import Llama
                    print("Attempting to create Llama instance with GPU layers...")
                    
                    # This should fail gracefully if GPU is not available
                    llm = Llama(
                        model_path=test_model_path,
                        n_gpu_layers=1,  # Just test with 1 layer
                        n_ctx=512,       # Small context for testing
                        verbose=True     # Enable verbose output
                    )
                    print("✓ GPU Test: SUCCESS - Llama instance created with GPU layers")
                    
                    # Clean up
                    del llm
                    
                except Exception as e:
                    print(f"✗ GPU Test: FAILED - {str(e)}")
            else:
                print(f"Test model not found: {test_model_path}")
                
        except Exception as e:
            print(f"GPU test error: {str(e)}")
            
    except ImportError as e:
        print(f"llama-cpp-python: IMPORT FAILED - {str(e)}")

def check_process_context():
    """Check process-specific context that might affect GPU access."""
    print_section("Process Context Check")
    
    # Check user context
    print(f"Current User: {os.environ.get('USERNAME', 'Unknown')}")
    print(f"User Domain: {os.environ.get('USERDOMAIN', 'Unknown')}")
    
    # Check if running in virtual environment
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        print(f"Virtual Environment: {venv_path}")
    else:
        print("Virtual Environment: Not detected")
    
    # Check system PATH for CUDA
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    cuda_paths = [p for p in path_dirs if 'cuda' in p.lower() or 'nvidia' in p.lower()]
    if cuda_paths:
        print("CUDA-related paths in PATH:")
        for path in cuda_paths:
            print(f"  {path}")
    else:
        print("No CUDA-related paths found in PATH")

def check_gpu_memory():
    """Check GPU memory usage."""
    print_section("GPU Memory Check")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                total, used, free = line.split(', ')
                print(f"GPU {i}: Total={total}MB, Used={used}MB, Free={free}MB")
        else:
            print("Failed to get GPU memory info")
    except Exception as e:
        print(f"Error checking GPU memory: {str(e)}")

def run_execution_context_test():
    """Test execution context (direct vs imported)."""
    print_section("Execution Context Test")
    
    context = sys.argv[0] if sys.argv else "Unknown"
    print(f"Script Context: {context}")
    
    # Check if we're being imported or run directly
    if __name__ == "__main__":
        print("Execution Mode: Direct execution (__main__)")
    else:
        print("Execution Mode: Imported module")
    
    # Check call stack depth
    import inspect
    stack = inspect.stack()
    print(f"Call Stack Depth: {len(stack)}")
    print("Call Stack:")
    for i, frame in enumerate(stack[:5]):  # Show first 5 frames
        print(f"  {i}: {frame.filename}:{frame.lineno} in {frame.function}")

def main():
    """Run all diagnostic checks."""
    print("GPU Environment Diagnostic Tool")
    print(f"Timestamp: {__import__('datetime').datetime.now()}")
    
    check_python_environment()
    check_cuda_environment()
    check_process_context()
    check_gpu_memory()
    check_llama_cpp_installation()
    run_execution_context_test()
    
    print_section("Summary")
    print("Diagnostic complete. Compare this output between:")
    print("1. Direct execution: python gpu_diagnostic.py")
    print("2. Flask execution: Import and call from server.py")
    print("\nLook for differences in:")
    print("- CUDA environment variables")
    print("- Process context")
    print("- GPU memory availability")
    print("- llama-cpp-python GPU test results")

if __name__ == "__main__":
    main()