#!/usr/bin/env python
"""
Setup script for the LLM Server environment.
This script helps set up the required dependencies and configuration.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60 + "\n")

def check_python_version():
    """Check if the Python version is adequate."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required.")
        print(f"Current Python version: {platform.python_version()}")
        sys.exit(1)
    print(f"✓ Python version {platform.python_version()} is adequate.")

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    if os.path.exists("venv"):
        print("✓ Virtual environment already exists.")
        return

    print("Creating a virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created successfully.")
    except subprocess.CalledProcessError:
        print("ERROR: Failed to create virtual environment.")
        sys.exit(1)

def install_dependencies():
    """Install required dependencies with CUDA support for llama-cpp-python."""
    print("Installing dependencies...")
    
    # Determine the pip executable based on the platform
    if platform.system() == "Windows":
        pip_path = os.path.join("venv", "Scripts", "pip")
    else:
        pip_path = os.path.join("venv", "bin", "pip")
    
    # Ensure pip is up to date
    try:
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    except subprocess.CalledProcessError:
        print("WARNING: Failed to upgrade pip. Continuing with installation...")
    
    # Install Flask and other base dependencies first
    base_packages = ["flask", "flask-cors", "psutil"]
    try:
        for package in base_packages:
            print(f"Installing {package}...")
            subprocess.run([pip_path, "install", package], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install base dependencies: {e}")
        sys.exit(1)
    
    # Install llama-cpp-python with CUDA support
    print("\nInstalling llama-cpp-python with CUDA support...")
    try:
        # First uninstall any existing llama-cpp-python package to avoid conflicts
        subprocess.run([pip_path, "uninstall", "-y", "llama-cpp-python"], check=False)
        subprocess.run([pip_path, "uninstall", "-y", "llama-cpp-python-cuda"], check=False)
        
        # Install with CUDA support
        print("Installing llama-cpp-python with CUDA...")
        env_vars = os.environ.copy()
        env_vars["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
        env_vars["FORCE_CMAKE"] = "1"
        
        install_command = [
            pip_path, "install", "llama-cpp-python", "--force-reinstall", "--no-cache-dir"
        ]
        
        subprocess.run(install_command, env=env_vars, check=True)
        print("✓ Successfully installed llama-cpp-python with CUDA support")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to install llama-cpp-python with CUDA support: {e}")
        print("Attempting to install pre-built llama-cpp-python-cuda package instead...")
        try:
            subprocess.run([pip_path, "install", "llama-cpp-python-cuda"], check=True)
            print("✓ Successfully installed pre-built llama-cpp-python-cuda package")
        except subprocess.CalledProcessError as e2:
            print(f"ERROR: Failed to install any version of llama-cpp-python with CUDA: {e2}")
            print("\nTroubleshooting tips:")
            print("1. Make sure you have CUDA toolkit installed")
            print("2. Make sure you have Visual Studio Build Tools installed with C++ support")
            print("3. Try running the command manually with administrator privileges")
            sys.exit(1)
            
    # Create requirements.txt for future use
    print("\nCreating requirements.txt file...")
    try:
        with open("requirements.txt", "w") as f:
            subprocess.run([pip_path, "freeze"], stdout=f, check=True)
        print("✓ Updated requirements.txt with installed packages")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Failed to create requirements.txt: {e}")

def check_model_paths():
    """Check if model paths in the configuration are valid."""
    print("Checking model paths in configuration...")
    
    # Load the configuration
    try:
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
            print("✓ Valid model paths found:")
            for name, path in valid_models:
                print(f"  - {name}: {path}")
        
        if invalid_models:
            print("WARNING: Invalid model paths found:")
            for name, path in invalid_models:
                print(f"  - {name}: {path}")
            
            print("\nYou may need to update the model paths in config.py.")
    except ImportError:
        print("WARNING: Could not import config.py. Make sure it exists and is properly formatted.")

def setup_log_directory():
    """Create a directory for logs if it doesn't exist."""
    print("Setting up log directory...")
    
    log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"✓ Created log directory at {log_dir}")
    else:
        print(f"✓ Log directory already exists at {log_dir}")

def create_gitignore():
    """Create a .gitignore file if it doesn't exist."""
    if os.path.exists(".gitignore"):
        print("✓ .gitignore file already exists.")
        return
        
    print("Creating .gitignore file...")
    
    gitignore_content = """# Virtual Environment
venv/
env/
ENV/

# Python cache files
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Distribution / packaging
dist/
build/
*.egg-info/

# Logs
logs/
*.log

# IDE specific files
.idea/
.vscode/
*.swp
*.swo

# OS specific files
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✓ Created .gitignore file.")

def main():
    """Main function to run all setup steps."""
    print_header("LLM Server Environment Setup")
    
    check_python_version()
    create_virtual_environment()
    install_dependencies()
    check_model_paths()
    setup_log_directory()
    create_gitignore()
    
    print_header("Setup Complete!")
    print("You can now run the server using:")
    print("  On Windows: start_server.bat")
    print("  On Linux/Mac: python server.py")
    print("\nMake sure your model paths in config.py are correct.")
    print("\nGitHub Best Practices:")
    print("1. This script created a .gitignore file to exclude venv from git")
    print("2. It also created/updated requirements.txt for dependency management")
    print("3. Use 'git add .', 'git commit', and 'git push' to push your code")
    print("\nFor other developers to use your code, they should:")
    print("1. Clone the repository")
    print("2. Run this setup script (python setup_environment.py)")
    print("3. Update config.py with their model paths")
    print("4. Run the server")

if __name__ == "__main__":
    main()