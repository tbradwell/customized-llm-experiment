#!/usr/bin/env python3
"""
Startup script for the Lawyer Contract Creation System.
This script handles environment setup, dependency checking, and server startup.
"""

import os
import sys
import subprocess
import logging
import signal
import time
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServerManager:
    """Manages the startup and shutdown of the contract generation server."""
    
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.project_root = Path(__file__).parent
        
    def check_environment(self) -> bool:
        """Check if the environment is properly configured."""
        logger.info("Checking environment configuration...")
        
        # Check if .env file exists
        env_file = self.project_root / ".env"
        if not env_file.exists():
            logger.error(".env file not found. Please create one from .env.example")
            return False
        
        # Check OpenAI API key
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            logger.error("OpenAI API key not configured. Please set OPENAI_API_KEY in .env file")
            return False
        
        # Check required directories
        required_dirs = ["data", "data/skeletons", "data/generated", "data/references"]
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Environment configuration is valid")
        return True
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'fastapi', 'uvicorn', 'python-docx', 'openai', 'nltk', 
            'rouge-score', 'scikit-learn', 'mlflow', 'spacy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.error("Install them with: pip install -r requirements.txt")
            return False
        
        # Check spaCy model
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Some features may be limited.")
            logger.warning("Install with: python -m spacy download en_core_web_sm")
        
        logger.info("All dependencies are available")
        return True
    
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        logger.info("Setting up MLflow tracking...")
        
        try:
            import mlflow
            from config.settings import settings
            
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            
            # Create experiment if it doesn't exist
            try:
                mlflow.create_experiment(settings.mlflow_experiment_name)
            except Exception:
                pass  # Experiment already exists
            
            logger.info("MLflow tracking setup completed")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {str(e)}")
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8000, 
                    reload: bool = False, workers: int = 1):
        """Start the FastAPI server."""
        logger.info(f"Starting server on {host}:{port}")
        
        # Build uvicorn command
        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", host,
            "--port", str(port),
            "--workers", str(workers)
        ]
        
        if reload:
            cmd.append("--reload")
        
        try:
            self.server_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                env=os.environ.copy()
            )
            
            logger.info(f"Server started with PID {self.server_process.pid}")
            logger.info(f"API documentation available at: http://localhost:{port}/docs")
            logger.info(f"Alternative docs at: http://localhost:{port}/redoc")
            
            # Wait for server to start
            time.sleep(2)
            
            return self.server_process
            
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            return None
    
    def stop_server(self):
        """Stop the server gracefully."""
        if self.server_process:
            logger.info("Stopping server...")
            self.server_process.terminate()
            
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, killing process...")
                self.server_process.kill()
                self.server_process.wait()
            
            logger.info("Server stopped")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop_server()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main function to start the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the Lawyer Contract Creation System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment checks")
    parser.add_argument("--setup-only", action="store_true", help="Only run setup, don't start server")
    
    args = parser.parse_args()
    
    server_manager = ServerManager()
    
    # Run setup checks
    if not args.skip_checks:
        logger.info("=== Lawyer Contract Creation System Startup ===")
        
        if not server_manager.check_environment():
            logger.error("Environment check failed. Exiting.")
            sys.exit(1)
        
        if not server_manager.check_dependencies():
            logger.error("Dependency check failed. Exiting.")
            sys.exit(1)
        
        server_manager.setup_mlflow()
        
        if args.setup_only:
            logger.info("Setup completed successfully!")
            return
    
    # Setup signal handlers
    server_manager.setup_signal_handlers()
    
    # Start server
    server_process = server_manager.start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )
    
    if server_process:
        try:
            # Wait for server to finish
            server_process.wait()
        except KeyboardInterrupt:
            pass
        finally:
            server_manager.stop_server()
    else:
        logger.error("Failed to start server")
        sys.exit(1)


if __name__ == "__main__":
    main()