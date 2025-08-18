#!/usr/bin/env python3
"""
Setup script for the Lawyer Contract Creation System environment.
This script sets up all necessary dependencies and configurations.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, description="Running command"):
    """Run a shell command and handle errors."""
    print(f"{description}: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"Python version {sys.version} is compatible")
    return True


def install_spacy_model():
    """Install spaCy English model."""
    print("Installing spaCy English model...")
    commands = [
        "python -m spacy download en_core_web_sm",
        "python -c \"import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded successfully')\""
    ]
    
    for cmd in commands:
        if not run_command(cmd, "Installing spaCy model"):
            print("Warning: spaCy model installation failed. Some features may be limited.")
            return False
    return True


def create_directories():
    """Create necessary directories."""
    print("Creating necessary directories...")
    directories = [
        "data",
        "data/skeletons", 
        "data/generated",
        "data/references",
        "examples/contracts",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def setup_environment_file():
    """Create .env file from example if it doesn't exist."""
    print("Setting up environment file...")
    
    env_file = Path(".env")
    env_example_file = Path(".env.example")
    
    if not env_file.exists() and env_example_file.exists():
        shutil.copy(".env.example", ".env")
        print("Created .env file from .env.example")
        print("Please edit .env file and add your OpenAI API key")
        return False
    elif env_file.exists():
        print(".env file already exists")
        return True
    else:
        print("Warning: No .env.example file found")
        return False


def verify_openai_key():
    """Verify OpenAI API key is set."""
    print("Verifying OpenAI API key...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            print("OpenAI API key is configured")
            return True
        else:
            print("Warning: Please set your OpenAI API key in the .env file")
            return False
    except ImportError:
        print("Warning: python-dotenv not installed")
        return False


def install_nltk_data():
    """Download required NLTK data."""
    print("Installing NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("NLTK data installed successfully")
        return True
    except ImportError:
        print("Warning: NLTK not installed")
        return False
    except Exception as e:
        print(f"Warning: NLTK data installation failed: {e}")
        return False


def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'python-docx', 'openai', 'nltk', 
        'rouge-score', 'scikit-learn', 'mlflow', 'spacy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("All required packages are installed")
    return True


def main():
    """Main setup function."""
    print("=== Lawyer Contract Creation System Setup ===\n")
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Check dependencies
    if not check_dependencies():
        success = False
    
    # Create directories
    create_directories()
    
    # Setup environment file
    if not setup_environment_file():
        success = False
    
    # Install NLTK data
    if not install_nltk_data():
        success = False
    
    # Install spaCy model
    if not install_spacy_model():
        success = False
    
    # Verify OpenAI key
    if not verify_openai_key():
        success = False
    
    print("\n=== Setup Summary ===")
    if success:
        print("✓ Environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your OpenAI API key")
        print("2. Run: python -m uvicorn src.api.main:app --reload")
        print("3. Visit: http://localhost:8000/docs for API documentation")
    else:
        print("⚠ Setup completed with warnings")
        print("Please address the warnings above before running the system")
    
    print("\nFor help, check the README.md file or visit the documentation")


if __name__ == "__main__":
    main()