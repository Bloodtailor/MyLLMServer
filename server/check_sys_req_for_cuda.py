#!/usr/bin/env python
"""
Standalone System Requirements Checker for LLM Server with CUDA Support
Run this to check if your system has everything needed for CUDA compilation.
"""

import subprocess
import os
import sys
import platform

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60 + "\n")

def print_section(text):
    """Print a section header."""
    print(f"\n{text}")
    print("-" * len(text))

def check_python_version():
    """Check Python version."""
    print_section("Python Version Check")
    
    version = sys.version_info
    print(f"Current Python version: {platform.python_version()}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print("✅ Python version is adequate")
        if version.minor == 11:
            print("✅ Using recommended Python 3.11")
        return True

def check_nvidia_gpu():
    """Check for NVIDIA GPU and drivers."""
    print_section("NVIDIA GPU Check")
    
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print("✅ NVIDIA GPU detected")
        
        # Parse GPU info from nvidia-smi output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line:
                # Extract GPU name
                parts = line.split('|')
                if len(parts) > 1:
                    gpu_info = parts[1].strip()
                    print(f"   GPU: {gpu_info}")
                break
        
        # Extract driver version
        for line in lines:
            if 'Driver Version:' in line:
                driver_part = line.split('Driver Version:')[1].split()[0]
                print(f"   Driver Version: {driver_part}")
                break
        
        return True
        
    except FileNotFoundError:
        print("❌ nvidia-smi command not found")
        print("   This means NVIDIA drivers are not installed or not in PATH")
        print("   Download drivers from: https://www.nvidia.com/drivers/")
        return False
    except subprocess.CalledProcessError:
        print("❌ nvidia-smi failed to run")
        print("   NVIDIA drivers may not be properly installed")
        return False

def check_cuda_toolkit():
    """Check for CUDA Toolkit installation."""
    print_section("CUDA Toolkit Check")
    
    cuda_found = False
    cuda_version = None
    
    # Method 1: Check CUDA_PATH environment variable
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.exists(cuda_path):
        print(f"✅ CUDA Toolkit found via CUDA_PATH: {cuda_path}")
        cuda_found = True
    
    # Method 2: Check nvcc command
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        print("✅ CUDA compiler (nvcc) is available")
        
        # Extract CUDA version from nvcc output
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                # Example: "Cuda compilation tools, release 12.1, V12.1.105"
                parts = line.split('release')[1].split(',')[0].strip()
                cuda_version = parts
                print(f"   CUDA Version: {cuda_version}")
                break
        
        cuda_found = True
    except FileNotFoundError:
        if not cuda_found:
            print("❌ nvcc command not found")
    except subprocess.CalledProcessError:
        if not cuda_found:
            print("❌ nvcc command failed")
    
    # Method 3: Check common installation directories
    if not cuda_found:
        common_cuda_paths = [
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
            "C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA",
            "/usr/local/cuda",
            "/opt/cuda"
        ]
        
        for path in common_cuda_paths:
            if os.path.exists(path):
                print(f"✅ CUDA Toolkit found at: {path}")
                cuda_found = True
                
                # Try to find version in subdirectories
                try:
                    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                    versions = [d for d in subdirs if d.startswith('v') and d[1:].replace('.', '').isdigit()]
                    if versions:
                        latest_version = sorted(versions)[-1]
                        print(f"   Latest version found: {latest_version}")
                except:
                    pass
                break
    
    if not cuda_found:
        print("❌ CUDA Toolkit not found")
        print("   Download from: https://developer.nvidia.com/cuda-toolkit")
        print("   Recommended: CUDA 11.8 or 12.x")
        
    return cuda_found

def check_visual_studio():
    """Check for Visual Studio Build Tools."""
    print_section("Visual Studio Build Tools Check")
    
    vs_found = False
    
    # Check for Visual Studio installations
    vs_paths = [
        ("Visual Studio 2022", "C:\\Program Files\\Microsoft Visual Studio\\2022"),
        ("Visual Studio 2019", "C:\\Program Files\\Microsoft Visual Studio\\2019"),
        ("Visual Studio 2019 (x86)", "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019"),
        ("Build Tools 2022", "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools"),
        ("Build Tools 2019", "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools"),
        ("Build Tools 2017", "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\BuildTools")
    ]
    
    found_installations = []
    
    for name, path in vs_paths:
        if os.path.exists(path):
            found_installations.append((name, path))
            vs_found = True
    
    if found_installations:
        print("✅ Visual Studio installations found:")
        for name, path in found_installations:
            print(f"   {name}: {path}")
            
            # Check for C++ build tools specifically
            cpp_tools_paths = [
                os.path.join(path, "VC", "Tools", "MSVC"),
                os.path.join(path, "Community", "VC", "Tools", "MSVC"),
                os.path.join(path, "Professional", "VC", "Tools", "MSVC"),
                os.path.join(path, "Enterprise", "VC", "Tools", "MSVC")
            ]
            
            for cpp_path in cpp_tools_paths:
                if os.path.exists(cpp_path):
                    print(f"      ✅ C++ tools found: {cpp_path}")
                    break
    else:
        print("❌ Visual Studio Build Tools not found")
        print("   Download 'Build Tools for Visual Studio' from:")
        print("   https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022")
        print("   ⚠️  Make sure to install the 'C++ build tools' workload!")
    
    return vs_found

def check_pip_and_venv():
    """Check pip and virtual environment capabilities."""
    print_section("Python Environment Check")
    
    # Check pip
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, check=True)
        print("✅ pip is available")
        print(f"   {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        print("   Install pip: python -m ensurepip --upgrade")
        return False
    
    # Check venv
    try:
        # Just check if venv module exists
        subprocess.run([sys.executable, "-m", "venv", "--help"], 
                      capture_output=True, check=True)
        print("✅ venv module is available")
    except subprocess.CalledProcessError:
        print("❌ venv module is not available")
        return False
    
    return True

def check_network_access():
    """Check if we can access PyPI for package installation."""
    print_section("Network Access Check")
    
    try:
        import urllib.request
        urllib.request.urlopen('https://pypi.org', timeout=10)
        print("✅ Can access PyPI (Python Package Index)")
        return True
    except:
        print("❌ Cannot access PyPI")
        print("   Check your internet connection")
        print("   You may need to configure proxy settings")
        return False

def main():
    """Main function to run all checks."""
    print_header("System Requirements Checker")
    print("Checking if your system can compile llama-cpp-python with CUDA support...")
    
    checks = [
        ("Python Version", check_python_version()),
        ("NVIDIA GPU", check_nvidia_gpu()),
        ("CUDA Toolkit", check_cuda_toolkit()),
        ("Visual Studio Build Tools", check_visual_studio()),
        ("Python Environment (pip/venv)", check_pip_and_venv()),
        ("Network Access", check_network_access())
    ]
    
    # Summary
    print_header("Summary")
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<30} {status}")
        if result:
            passed_checks += 1
    
    print(f"\nPassed: {passed_checks}/{total_checks} checks")
    
    # Recommendations
    if passed_checks == total_checks:
        print("\n🎉 All checks passed! Your system should be able to compile llama-cpp-python with CUDA support.")
        print("\nRecommended installation order:")
        print("1. pip install llama-cpp-python-cuda  (try this first)")
        print("2. If that fails, try building from source with CUDA flags")
    elif passed_checks >= 4:  # Most important checks passed
        print("\n⚠️  Most checks passed. CUDA compilation might work, but some issues detected.")
        print("You can try installing, but it might fall back to CPU-only version.")
    else:
        print("\n❌ Several critical checks failed. CUDA compilation will likely fail.")
        print("You can still use the CPU-only version of llama-cpp-python.")
    
    # Specific recommendations based on failures
    print("\n" + "="*60)
    print("INSTALLATION RECOMMENDATIONS")
    print("="*60)
    
    failed_checks = [(name, result) for name, result in checks if not result]
    
    if not any(name == "NVIDIA GPU" and not result for name, result in checks):
        if any(name == "CUDA Toolkit" and not result for name, result in checks):
            print("\n🔧 Missing CUDA Toolkit:")
            print("   1. Go to https://developer.nvidia.com/cuda-toolkit")
            print("   2. Download CUDA 11.8 or 12.x (match your driver version)")
            print("   3. Run the installer and follow the setup wizard")
    
        if any(name == "Visual Studio Build Tools" and not result for name, result in checks):
            print("\n🔧 Missing Visual Studio Build Tools:")
            print("   1. Go to https://visualstudio.microsoft.com/downloads/")
            print("   2. Download 'Build Tools for Visual Studio 2022'")
            print("   3. During installation, select 'C++ build tools' workload")
            print("   4. This installs the compiler needed for building packages")
    
    print(f"\n📊 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {platform.python_version()}")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()