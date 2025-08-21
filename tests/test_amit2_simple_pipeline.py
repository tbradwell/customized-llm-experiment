#!/usr/bin/env python3
"""
Document Generation Script - No Evaluation
Generate contract documents without running evaluation metrics.

Usage: python generate_document_only.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append('.')

from src.core.simple_pipeline import SimplePipeline
from src.utils.error_handler import ProcessingError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentOnlyPipeline(SimplePipeline):
    """Extended SimplePipeline that skips evaluation."""
    
    def _run_evaluation(self, final_text: str, output_dir: str, artifacts_info: dict) -> dict:
        """Override evaluation to skip it entirely."""
        logger.info("Skipping evaluation as requested - document generation only")
        return {}


def main():
    """Main function to generate document without evaluation."""
    
    # Configuration
    NEW_DATA_DIR = "examples/amit_test2/new_data"
    SKELETON_PATH = "examples/amit_test2/skeleton_oracle.docx"
    OUTPUT_DIR = "experiments/amit_test2_generation_only"
    CONTRACT_TYPE = "legal_claim"
    TEST_TYPE = "generation_only"
    
    logger.info("🚀 Starting Document Generation (No Evaluation)")
    logger.info(f"New data directory: {NEW_DATA_DIR}")
    logger.info(f"Skeleton file: {SKELETON_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Verify input files exist
    skeleton_path = Path(SKELETON_PATH)
    new_data_path = Path(NEW_DATA_DIR)
    
    if not skeleton_path.exists():
        logger.error(f"❌ Skeleton file not found: {SKELETON_PATH}")
        return 1
    
    if not new_data_path.exists():
        logger.error(f"❌ New data directory not found: {NEW_DATA_DIR}")
        return 1
    
    # List available data files
    data_files = list(new_data_path.glob("*"))
    logger.info(f"📁 Found {len(data_files)} data files:")
    for file in data_files:
        logger.info(f"  • {file.name}")
    
    try:
        # Initialize pipeline
        logger.info("🔧 Initializing document generation pipeline...")
        pipeline = DocumentOnlyPipeline()
        
        # Run pipeline without evaluation
        logger.info("⚙️ Starting document generation...")
        results = pipeline.run_pipeline(
            new_data_dir=NEW_DATA_DIR,
            skeleton_path=SKELETON_PATH,
            output_dir=OUTPUT_DIR,
            contract_type=CONTRACT_TYPE,
            test_type=TEST_TYPE
        )
        
        # Report results
        if results.get('success'):
            logger.info("✅ Document generation completed successfully!")
            logger.info(f"📊 Generation Statistics:")
            logger.info(f"  • Initial placeholders: {results.get('initial_placeholders', 'N/A')}")
            logger.info(f"  • Final placeholders: {results.get('final_placeholders', 'N/A')}")
            logger.info(f"  • Completion iterations: {results.get('completion_iterations', 'N/A')}")
            logger.info(f"  • Completion status: {results.get('completion_status', 'N/A')}")
            logger.info(f"  • Final content length: {results.get('final_content_length', 'N/A')} characters")
            
            # Show artifact locations
            artifacts = results.get('artifacts', {})
            if artifacts:
                logger.info(f"📄 Generated files:")
                for artifact_type, path in artifacts.items():
                    logger.info(f"  • {artifact_type}: {path}")
            
            logger.info(f"🎯 MLflow Run ID: {results.get('run_id', 'N/A')}")
            
        else:
            logger.error("❌ Document generation failed")
            return 1
            
    except ProcessingError as e:
        logger.error(f"❌ Processing error: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1
    
    logger.info("🏁 Document generation completed")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)