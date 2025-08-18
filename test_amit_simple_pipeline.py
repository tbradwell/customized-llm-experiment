#!/usr/bin/env python3
"""Test script for simple pipeline without past examples."""

import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.core.simple_pipeline import SimplePipeline

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    """Run the simple pipeline test."""
    setup_logging()
    
    print("🚀 AMIT TEST - Simple Pipeline (No Past Examples)")
    print("=" * 60)
    
    try:
        # Set up paths
        new_data_dir = "examples/amit_test/new_data"
        skeleton_path = "examples/amit_test/sekeleton_oracle.docx"
        output_dir = "experiments/amit_simple_pipeline"
        
        print(f"📁 New data dir: {new_data_dir}")
        print(f"📝 Skeleton: {skeleton_path}")
        print(f"🗂️  Output dir: {output_dir}")
        
        # Initialize and run pipeline
        pipeline = SimplePipeline()
        
        print(f"\n🔄 Running pipeline...")
        print("-" * 40)
        
        results = pipeline.run_pipeline(
            new_data_dir=new_data_dir,
            skeleton_path=skeleton_path,
            output_dir=output_dir,
            contract_type="legal_claim",
            test_type="amit_simple"
        )
        
        # Display results
        print(f"\n📊 PIPELINE RESULTS")
        print("=" * 40)
        print(f"✅ Success: {results['success']}")
        print(f"🆔 MLflow Run ID: {results['run_id']}")
        print(f"📝 Initial placeholders: {results['initial_placeholders']}")
        print(f"📝 Final placeholders: {results['final_placeholders']}")
        print(f"🔄 Completion iterations: {results['completion_iterations']}")
        print(f"✅ Completion status: {results['completion_status']}")
        print(f"📏 Final content length: {results['final_content_length']} characters")
        
        # Show evaluation results if available
        if results['evaluation_results']:
            print(f"\n📊 EVALUATION METRICS:")
            for metric_name, score in results['evaluation_results'].items():
                print(f"  • {metric_name.upper()}: {score:.3f}")
        
        # Show artifacts
        print(f"\n📁 ARTIFACTS SAVED:")
        for artifact_type, path in results['artifacts'].items():
            print(f"  • {artifact_type}: {path}")
        
        # Success assessment
        if results['final_placeholders'] == 0:
            print(f"\n🎉 PERFECT SUCCESS! 🎉")
            print("=" * 30)
            print("✅ All placeholders filled successfully!")
            print(f"📂 Check MLflow UI: mlruns/0/{results['run_id']}/")
        else:
            print(f"\n⚠️  Some placeholders remain unfilled")
            print(f"📂 Check MLflow UI for details: mlruns/0/{results['run_id']}/")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)