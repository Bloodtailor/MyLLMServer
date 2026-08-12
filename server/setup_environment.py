#!/usr/bin/env python
"""
Complete setup script for the LLM Server environment.
Checks system requirements and sets up the environment with CUDA support.
"""

import os
import sys
import platform
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Prebuilt CUDA wheels published by llama-cpp-python's author
CUDA_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"

def venv_bin(executable):
    """Path to an executable inside the local venv."""
    subdir = "Scripts" if platform.system() == "Windows" else "bin"
    return os.path.join(SCRIPT_DIR, "venv", subdir, executable)

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
    """Check if the Python version is adequate."""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required.")
        print(f"Current Python version: {platform.python_version()}")
        return False
    
    print(f"✅ Python version {platform.python_version()} is adequate.")
    if version.minor == 11:
        print("✅ Using recommended Python 3.11")
    return True

def check_nvidia_gpu():
    """Check for NVIDIA GPU and drivers."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print("✅ NVIDIA GPU detected")
        
        # Parse GPU info
        lines = result.stdout.split('\n')
        for line in lines:
            if any(gpu in line for gpu in ['GeForce', 'RTX', 'GTX', 'Quadro']):
                parts = line.split('|')
                if len(parts) > 1:
                    gpu_info = parts[1].strip()
                    print(f"   GPU: {gpu_info}")
                break
        return True
        
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ NVIDIA GPU or drivers not detected")
        print("   Install NVIDIA GPU drivers from https://www.nvidia.com/drivers/")
        return False

def check_cuda_toolkit():
    """Check for CUDA Toolkit installation."""
    cuda_found = False
    
    # Check CUDA_PATH environment variable
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.exists(cuda_path):
        print(f"✅ CUDA Toolkit found: {cuda_path}")
        cuda_found = True
    
    # Check nvcc command
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        print("✅ CUDA compiler (nvcc) available")
        
        # Extract version
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                parts = line.split('release')[1].split(',')[0].strip()
                print(f"   CUDA Version: {parts}")
                break
        cuda_found = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    # Check common paths
    if not cuda_found:
        common_paths = [
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
            "C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                print(f"✅ CUDA Toolkit found at: {path}")
                cuda_found = True
                break
    
    if not cuda_found:
        print("❌ CUDA Toolkit not found")
        print("   Download from: https://developer.nvidia.com/cuda-toolkit")
    
    return cuda_found

def check_visual_studio():
    """Check for Visual Studio Build Tools."""
    vs_paths = [
        "C:\\Program Files\\Microsoft Visual Studio\\2022",
        "C:\\Program Files\\Microsoft Visual Studio\\2019",
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools",
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools"
    ]
    
    for path in vs_paths:
        if os.path.exists(path):
            print(f"✅ Visual Studio/Build Tools found: {path}")
            return True
    
    print("❌ Visual Studio Build Tools not found")
    print("   Download from: https://visualstudio.microsoft.com/downloads/")
    print("   Install 'C++ build tools' workload")
    return False

def check_pip_and_network():
    """Check pip and network access."""
    # Check pip
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ pip available: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("❌ pip not available")
        return False
    
    # Check network
    try:
        import urllib.request
        urllib.request.urlopen('https://pypi.org', timeout=10)
        print("✅ Network access to PyPI confirmed")
        return True
    except:
        print("❌ Cannot access PyPI - check internet connection")
        return False

def run_system_requirements_check():
    """Run comprehensive system requirements check."""
    print_header("System Requirements Check")
    print("Checking if your system can compile llama-cpp-python with CUDA support...")
    
    checks = [
        ("Python Version", check_python_version()),
        ("NVIDIA GPU", check_nvidia_gpu()),
        ("CUDA Toolkit", check_cuda_toolkit()),
        ("Visual Studio Build Tools", check_visual_studio()),
        ("pip & Network", check_pip_and_network())
    ]
    
    passed_checks = sum(1 for _, result in checks if result)
    total_checks = len(checks)
    
    print_section("Requirements Summary")
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<25} {status}")
    
    print(f"\nPassed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("\n🎉 All requirements met! CUDA compilation should work.")
        return True
    elif passed_checks >= 3:
        print("\n⚠️  Some requirements missing. Will try CUDA but may fall back to CPU.")
        choice = input("Continue anyway? (y/n): ")
        return choice.lower() == 'y'
    else:
        print("\n❌ Critical requirements missing. Setup will likely fail.")
        choice = input("Continue anyway? (y/n): ")
        return choice.lower() == 'y'

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    venv_dir = os.path.join(SCRIPT_DIR, "venv")
    if os.path.exists(venv_dir):
        print("✅ Virtual environment already exists.")
        return

    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        print("✅ Virtual environment created successfully.")
    except subprocess.CalledProcessError:
        print("ERROR: Failed to create virtual environment.")
        sys.exit(1)

def install_dependencies():
    """Install required dependencies with CUDA support for llama-cpp-python."""
    print_section("Installing Dependencies")
    
    pip_path = venv_bin("pip")
    python_path = venv_bin("python")

    # Upgrade pip first
    try:
        print("Upgrading pip...")
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        print("✅ pip upgraded successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Failed to upgrade pip, continuing...")
    
    # Install base packages
    base_packages = ["flask", "flask-cors", "psutil"]
    try:
        for package in base_packages:
            print(f"Installing {package}...")
            subprocess.run([pip_path, "install", package], check=True)
        print("✅ Base packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install base dependencies: {e}")
        sys.exit(1)
    
    # Install llama-cpp-python with CUDA support
    print_section("Installing llama-cpp-python with CUDA Support")

    # Uninstall any existing build so a CPU wheel can't shadow a CUDA one
    subprocess.run([pip_path, "uninstall", "-y", "llama-cpp-python"], check=False)

    # Method 1: Prebuilt CUDA wheels (fastest, no compiler needed)
    print("Method 1: Trying the prebuilt CUDA wheel index (cu124)...")
    try:
        subprocess.run([
            pip_path, "install", "llama-cpp-python",
            "--extra-index-url", CUDA_WHEEL_INDEX
        ], check=True)
        print("Installed a wheel from the CUDA index, verifying it...")
        # pip's exit code only says "something installed" - --extra-index-url
        # merges with PyPI, so it can hand back a CPU-only wheel and still
        # succeed. Only a build that reports GPU offload ends the ladder here.
        if report_gpu_support(python_path):
            print("🎉 Successfully installed a prebuilt CUDA wheel!")
            return
        print("❌ That wheel has no GPU offload (likely resolved from PyPI), trying Method 2...")
    except subprocess.CalledProcessError:
        print("❌ No matching CUDA wheel, trying Method 2...")

    # Method 2: Build from source with CUDA flags
    print("\nMethod 2: Building from source with CUDA support...")
    try:
        env_vars = os.environ.copy()
        env_vars["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        env_vars["FORCE_CMAKE"] = "1"

        # --no-binary forces an actual compile; without it pip happily reuses a
        # prebuilt CPU wheel and cmake never runs.
        install_command = [
            pip_path, "install", "llama-cpp-python",
            "--force-reinstall", "--no-cache-dir",
            "--no-binary", "llama-cpp-python"
        ]

        print("This may take 5-10 minutes to compile...")
        subprocess.run(install_command, env=env_vars, check=True)
        print("Source build finished, verifying it...")
        if report_gpu_support(python_path):
            print("🎉 Successfully built llama-cpp-python with CUDA support!")
            return
        print("❌ The source build has no GPU offload, falling back to CPU version...")
    except subprocess.CalledProcessError:
        print("❌ Source build failed, falling back to CPU version...")

    # Fallback: CPU-only version
    print("\nFallback: Installing CPU-only version...")
    try:
        subprocess.run([pip_path, "install", "llama-cpp-python"], check=True)
        print("✅ Installed CPU-only version as fallback")
        print("⚠️  Your models will run on CPU only (slower but still works)")
        report_gpu_support(python_path)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Even CPU installation failed: {e}")
        sys.exit(1)

def report_gpu_support(python_path):
    """Report whether the installed llama-cpp-python can actually offload to the GPU.

    Returns True only when the installed build reports GPU offload support, so
    callers can use it to decide whether to keep walking the fallback ladder.
    """
    try:
        result = subprocess.run(
            [python_path, "-c",
             "import llama_cpp; print(bool(llama_cpp.llama_supports_gpu_offload()))"],
            capture_output=True, text=True, check=True
        )
        if result.stdout.strip() == "True":
            print("✅ Verified: this build supports GPU offload")
            return True
        print("⚠️  This build does NOT support GPU offload - models will run on CPU")
        return False
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Could not verify GPU offload support: {e}")
        return False

def check_model_paths():
    """Check if model paths in the configuration are valid."""
    print_section("Checking Model Configuration")

    try:
        sys.path.insert(0, SCRIPT_DIR)
        from config import MODEL_ASSIGNMENTS
        
        valid_models = []
        invalid_models = []
        
        for model_name, model_config in MODEL_ASSIGNMENTS.items():
            model_path = model_config.get("model_path", "")
            if os.path.exists(model_path):
                valid_models.append((model_name, model_path))
            else:
                invalid_models.append((model_name, model_path))
        
        if valid_models:
            print("✅ Valid model paths found:")
            for name, path in valid_models:
                print(f"   {name}: {os.path.basename(path)}")
        
        if invalid_models:
            print("⚠️  Invalid model paths (update config.py):")
            for name, path in invalid_models:
                print(f"   {name}: {path}")
    
    except ImportError:
        print("⚠️  config.py not found - you'll need to configure model paths")

def setup_log_directory():
    """Create a directory for logs if it doesn't exist."""
    log_dir = os.path.join(SCRIPT_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"✅ Created log directory: {log_dir}")
    else:
        print(f"✅ Log directory exists: {log_dir}")

def create_lock_file():
    """Snapshot the installed packages next to the curated requirements.txt."""
    print("Writing requirements.lock.txt...")

    lock_path = os.path.join(SCRIPT_DIR, "requirements.lock.txt")
    try:
        with open(lock_path, "w") as f:
            subprocess.run([venv_bin("pip"), "freeze"], stdout=f, check=True)
        print(f"✅ Wrote {lock_path} (requirements.txt is hand-curated and left alone)")
    except subprocess.CalledProcessError:
        print("⚠️  Failed to write requirements.lock.txt")

def main():
    """Main function to run all setup steps."""
    print_header("LLM Server Environment Setup")
    print("This script will set up your environment for running LLM models with CUDA support.")
    
    # Run system requirements check first
    if not run_system_requirements_check():
        print("Setup cancelled by user.")
        sys.exit(1)
    
    # Proceed with setup
    create_virtual_environment()
    install_dependencies()
    check_model_paths()
    setup_log_directory()
    create_lock_file()
    
    print_header("Setup Complete!")
    print("🎉 Environment setup finished successfully!")
    print("\nNext steps:")
    print("1. Update model paths in config.py if needed")
    print("2. Run the server using: start_server.bat")
    print("3. Configure your Android app with the server IP address")
    
    print(f"\n📋 Your setup summary:")
    print(f"   Python: {platform.python_version()}")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Virtual environment: Created in ./venv/")
    print(f"   Log directory: ./logs/")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()