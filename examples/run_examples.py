#!/usr/bin/env python3
"""
Master script to run all examples for the Lawyer Contract Creation System.
This script demonstrates the complete functionality of the system.
"""

import sys
import subprocess
import time
from pathlib import Path


def run_script(script_path: str, description: str):
    """Run a Python script and handle its output."""
    print(f"\n{'='*80}")
    print(f"🚀 Running: {description}")
    print(f"📄 Script: {script_path}")
    print(f"{'='*80}")
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Script completed successfully!")
            if result.stdout:
                print("\n📤 Output:")
                print(result.stdout)
        else:
            print("❌ Script failed!")
            if result.stderr:
                print("\n🚨 Error output:")
                print(result.stderr)
            if result.stdout:
                print("\n📤 Standard output:")
                print(result.stdout)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Script timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False


def check_server_running():
    """Check if the API server is running."""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def start_server():
    """Start the API server."""
    print("🌐 Starting API server...")
    print("💡 Note: You may need to start the server manually with:")
    print("   python start_server.py")
    print("   Or: uvicorn src.api.main:app --reload")
    print("\n⏳ Waiting for server to be available...")
    
    # Wait for server to be available (user should start it manually)
    for i in range(30):  # Wait up to 30 seconds
        if check_server_running():
            print("✅ Server is running and accessible!")
            return True
        time.sleep(1)
        print(f"   Checking... ({i+1}/30)")
    
    print("❌ Server is not accessible. Please start it manually.")
    return False


def main():
    """Run all examples in sequence."""
    print("🏛️ Lawyer Contract Creation System - Complete Examples Suite")
    print("="*80)
    
    examples_dir = Path(__file__).parent
    examples = [
        {
            "script": "create_sample_docx.py",
            "description": "Create Sample .docx Files",
            "requires_server": False
        },
        {
            "script": "quality_analysis_example.py", 
            "description": "Quality Metrics Analysis",
            "requires_server": False
        },
        {
            "script": "mlflow_analysis_example.py",
            "description": "MLflow Experiment Analysis", 
            "requires_server": False
        },
        {
            "script": "api_client_example.py",
            "description": "API Client Examples",
            "requires_server": True
        }
    ]
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    
    # Check if required files exist
    required_files = [
        "examples/example_contract_data.json",
        "data/skeletons/service_agreement_skeleton.txt",
        "data/skeletons/nda_skeleton.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Please run the setup first:")
        print("   python setup_environment.py")
        return
    
    print("✅ All required files found")
    
    # Run examples
    success_count = 0
    server_started = False
    
    for example in examples:
        script_path = examples_dir / example["script"]
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            continue
        
        # Check if server is required
        if example["requires_server"] and not server_started:
            if not check_server_running():
                print(f"\n⚠️ {example['description']} requires the API server to be running")
                if not start_server():
                    print(f"⏭️ Skipping {example['description']} - server not available")
                    continue
            server_started = True
        
        # Run the example
        success = run_script(str(script_path), example["description"])
        if success:
            success_count += 1
        
        # Small delay between examples
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 EXAMPLES EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {success_count}/{len(examples)}")
    
    if success_count == len(examples):
        print("🎉 All examples completed successfully!")
        
        print(f"\n📁 Generated Files:")
        print("• Contract files: generated_*.docx")
        print("• Quality visualizations: *_quality_analysis.png")
        print("• MLflow tracking: mlflow.db (if configured)")
        
        print(f"\n💡 Next Steps:")
        print("• Review generated contracts for quality")
        print("• Check MLflow UI for experiment tracking: mlflow ui")
        print("• Explore the API documentation: http://localhost:8000/docs")
        print("• Run your own contract generation workflows")
        
    else:
        failed = len(examples) - success_count
        print(f"⚠️ {failed} example(s) failed - check the output above for details")
        
        print(f"\n🔧 Troubleshooting:")
        print("• Ensure all dependencies are installed: pip install -r requirements.txt")
        print("• Run setup script: python setup_environment.py")
        print("• Start API server: python start_server.py")
        print("• Check OpenAI API key in .env file")
    
    print(f"\n🏁 Examples suite completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)