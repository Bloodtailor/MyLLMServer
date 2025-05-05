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
    """Install required dependencies."""
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
    
    # Check if requirements.txt exists
    if os.path.exists("requirements.txt"):
        print("Installing dependencies from requirements.txt...")
        try:
            subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
            print("✓ All dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install dependencies: {e}")
            sys.exit(1)
    else:
        # Install individual packages if requirements.txt doesn't exist
        required_packages = ["flask", "flask-cors", "llama-cpp-python", "psutil"]
        
        try:
            for package in required_packages:
                print(f"Installing {package}...")
                subprocess.run([pip_path, "install", package], check=True)
            
            # Create requirements.txt for future use
            print("Creating requirements.txt file...")
            with open("requirements.txt", "w") as f:
                subprocess.run([pip_path, "freeze"], stdout=f, check=True)
            
            print("✓ All dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install dependencies: {e}")
            sys.exit(1)

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