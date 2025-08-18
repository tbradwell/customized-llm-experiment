#!/usr/bin/env python3
"""
Command-line script for running the complete quality pipeline with MLflow tracking.

This script allows users to run the full contract generation pipeline from the command line,
providing skeleton files, contract data, and configuration parameters to generate 
high-quality contracts with comprehensive evaluation and MLflow experiment tracking.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.quality_pipeline import QualityAssurancePipeline
from src.utils.data_reader import DataReader
from src.utils.error_handler import ProcessingError, error_handler
from config.settings import settings

# CONFIGURATION CONSTANTS
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_EXPERIMENT_NAME = "contract_generation"
MAX_SKELETON_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_DATA_FILE_SIZE = 10 * 1024 * 1024      # 10MB

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_file_path(file_path: str, max_size: int, description: str) -> Path:
    """Validate that a file exists and is within size limits."""
    path = Path(file_path)
    
    if not path.exists():
        raise ValueError(f"{description} file not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"{description} path is not a file: {file_path}")
    
    file_size = path.stat().st_size
    if file_size > max_size:
        raise ValueError(f"{description} file too large: {file_size / (1024*1024):.1f}MB (max {max_size / (1024*1024)}MB)")
    
    return path


def load_contract_data(data_input: str) -> Dict[str, Any]:
    """Load contract data from file or directory containing multiple files."""
    try:
        data_path = Path(data_input)
        
        if data_path.is_file():
            # Single file processing
            return load_single_contract_data(data_input)
        elif data_path.is_dir():
            # Directory processing - combine all files
            return load_directory_contract_data(data_input)
        else:
            raise ValueError(f"Data input path does not exist: {data_input}")
        
    except ProcessingError:
        raise
    except Exception as e:
        raise ValueError(f"Error loading contract data: {e}")


def load_single_contract_data(data_file: str) -> Dict[str, Any]:
    """Load contract data from a single file."""
    try:
        data_reader = DataReader()
        data = data_reader.read_contract_data(data_file)
        
        # Validate that we got some data
        if not data:
            raise ValueError("No data extracted from file")
        
        # Check for recommended fields
        required_fields = ["client_name", "provider_name"]
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        
        if missing_fields:
            logger.warning(f"Missing recommended fields: {', '.join(missing_fields)}")
            
            # If raw content exists, suggest manual data entry
            if "raw_content" in data:
                logger.info("Raw content detected. You may need to provide structured data manually.")
        
        return data
        
    except ProcessingError:
        raise
    except Exception as e:
        raise ValueError(f"Error loading contract data from file: {e}")


def load_directory_contract_data(data_dir: str) -> Dict[str, Any]:
    """Load and combine contract data from all files in a directory."""
    try:
        data_reader = DataReader()
        dir_path = Path(data_dir)
        
        # Get supported file extensions
        supported_formats = data_reader.get_supported_formats()
        all_extensions = set()
        for format_list in supported_formats.values():
            all_extensions.update(format_list)
        
        # Find all supported files in directory
        data_files = []
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in all_extensions:
                data_files.append(file_path)
        
        if not data_files:
            raise ValueError(f"No supported data files found in directory: {data_dir}")
        
        logger.info(f"Found {len(data_files)} data files in directory")
        
        # Process each file and combine data
        combined_data = {}
        raw_contents = []
        all_metadata = []
        
        for i, file_path in enumerate(sorted(data_files)):
            logger.info(f"Processing file {i+1}/{len(data_files)}: {file_path.name}")
            
            try:
                file_data = data_reader.read_contract_data(str(file_path))
                
                # Extract metadata
                if "_metadata" in file_data:
                    all_metadata.append({
                        "file": file_path.name,
                        "metadata": file_data["_metadata"]
                    })
                    del file_data["_metadata"]
                
                # Handle raw content
                if "raw_content" in file_data:
                    raw_contents.append(f"=== {file_path.name} ===\n{file_data['raw_content']}")
                    del file_data["raw_content"]
                
                # Merge structured data (prefer non-empty values)
                for key, value in file_data.items():
                    if value and (key not in combined_data or not combined_data[key]):
                        combined_data[key] = value
                
            except Exception as e:
                logger.warning(f"Could not process file {file_path.name}: {e}")
                continue
        
        # ENHANCED: Always include raw content for AI processing
        if raw_contents:
            combined_data["source_documents"] = "\n\n".join(raw_contents)
        
        # Add structured summary for better AI processing
        combined_data = enhance_contract_data(combined_data, raw_contents)
        
        # Add directory metadata
        combined_data["_metadata"] = {
            "source_type": "directory",
            "source_path": str(dir_path),
            "processed_files": len(data_files),
            "file_metadata": all_metadata,
            "extraction_method": "Multi-file directory processing"
        }
        
        logger.info(f"Combined data from {len(data_files)} files")
        return combined_data
        
    except Exception as e:
        raise ValueError(f"Error loading directory contract data: {e}")


def enhance_contract_data(combined_data: Dict[str, Any], raw_contents: List[str]) -> Dict[str, Any]:
    """Enhance contract data with AI-friendly summaries and extracted information."""
    
    # If we have rich content but minimal structured data, add helpful summaries
    if raw_contents and len(combined_data) < 5:
        combined_data["content_summary"] = f"Extracted content from {len(raw_contents)} documents including correspondence, legal documents, and contract-related materials."
        
        # Add hints for AI processing
        combined_data["processing_instructions"] = {
            "use_source_documents": "The 'source_documents' field contains the full extracted text from all input files. Use this content to inform contract generation.",
            "extract_parties": "Identify party names, addresses, and contact information from the source documents.",
            "extract_terms": "Extract relevant contract terms, dates, amounts, and conditions from the source documents.",
            "context": "This appears to be legal correspondence and documentation that should inform the contract generation."
        }
    
    return combined_data


def load_reference_contracts(reference_files: Optional[str]) -> Optional[list]:
    """Load reference contracts from files."""
    if not reference_files:
        return None
    
    reference_contracts = []
    
    for file_path in reference_files.split(','):
        file_path = file_path.strip()
        try:
            ref_path = validate_file_path(file_path, MAX_SKELETON_FILE_SIZE, "Reference contract")
            
            # For now, assume text files - in real implementation would handle .docx
            with open(ref_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    reference_contracts.append(content)
                    
        except Exception as e:
            logger.warning(f"Could not load reference contract {file_path}: {e}")
    
    return reference_contracts if reference_contracts else None


def create_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def create_experiment_folder(experiment_name: str, skeleton_path: str, data_path: str) -> Path:
    """Create organized experiment folder structure."""
    import time
    
    # Create experiment folder with timestamp
    timestamp = int(time.time())
    exp_folder = Path(f"experiments/{experiment_name}_{timestamp}")
    
    # Create folder structure
    (exp_folder / "inputs").mkdir(parents=True, exist_ok=True)
    (exp_folder / "outputs").mkdir(parents=True, exist_ok=True)
    (exp_folder / "reports").mkdir(parents=True, exist_ok=True)
    (exp_folder / "logs").mkdir(parents=True, exist_ok=True)
    
    # Copy input files
    import shutil
    skeleton_file = Path(skeleton_path)
    if skeleton_file.exists():
        shutil.copy2(skeleton_file, exp_folder / "inputs" / "skeleton.docx")
    
    # Copy or reference data files
    data_file = Path(data_path)
    if data_file.is_file():
        shutil.copy2(data_file, exp_folder / "inputs" / f"data{data_file.suffix}")
    elif data_file.is_dir():
        shutil.copytree(data_file, exp_folder / "inputs" / "data", dirs_exist_ok=True)
    
    # Create experiment metadata
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "skeleton_path": skeleton_path,
        "data_path": data_path,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(exp_folder / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Created experiment folder: {exp_folder}")
    return exp_folder


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete quality pipeline for contract generation with MLflow tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with JSON data
  python scripts/run_pipeline.py --skeleton contract_skeleton.docx --data contract_data.json

  # Using PDF data file
  python scripts/run_pipeline.py --skeleton skeleton.docx --data contract_info.pdf

  # Using image (OCR) data extraction
  python scripts/run_pipeline.py --skeleton skeleton.docx --data scanned_contract.png

  # Using email data (.msg or .eml)
  python scripts/run_pipeline.py --skeleton skeleton.docx --data client_request.msg

  # Processing entire directory of data files (combines all supported files)
  python scripts/run_pipeline.py --skeleton data/skeletons/skeleton_oracle.docx --data examples/amit_test/new_data/

  # Real example with your data
  python scripts/run_pipeline.py --skeleton data/skeletons/skeleton_oracle.docx --data examples/amit_test/new_data/ --experiment "amit_test_case" --verbose

  # With output file and checklist
  python scripts/run_pipeline.py --skeleton skeleton.docx --data data.docx --output final_contract.docx --checklist "Include payment terms,Add confidentiality clause"

  # With reference contracts and custom experiment name
  python scripts/run_pipeline.py --skeleton skeleton.docx --data data.pdf --references "ref1.txt,ref2.txt" --experiment "nda_generation_v2"

  # Disable MLflow tracking
  python scripts/run_pipeline.py --skeleton skeleton.docx --data data.json --no-mlflow

Supported Data Formats:
  • JSON: Structured data file (.json)
  • Documents: PDF (.pdf), Word (.doc, .docx), Text (.txt)
  • Images: PNG, JPG, TIFF, BMP (.png, .jpg, .jpeg, .tiff, .bmp) - uses OCR
  • Emails: Outlook (.msg), Standard (.eml)
  • Directories: Process all supported files in a directory and combine data

Contract Data Structure (JSON example):
  {
    "client_name": "Example Corp",
    "provider_name": "Service Provider LLC",
    "contract_type": "service_agreement",
    "contract_value": "$50,000",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    ...
  }
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--skeleton', 
        required=True,
        help='Path to the contract skeleton .docx file'
    )
    
    parser.add_argument(
        '--data', 
        required=True,
        help='Path to contract data file or directory (supports JSON, PDF, DOC/DOCX, images, emails)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output',
        help='Output path for the generated contract (default: auto-generated in outputs/)'
    )
    
    parser.add_argument(
        '--checklist',
        help='Comma-separated list of requirements to check (e.g., "Include payment terms,Add signatures")'
    )
    
    parser.add_argument(
        '--references',
        help='Comma-separated list of reference contract files for quality comparison'
    )
    
    parser.add_argument(
        '--experiment',
        default=DEFAULT_EXPERIMENT_NAME,
        help=f'MLflow experiment name (default: {DEFAULT_EXPERIMENT_NAME})'
    )
    
    parser.add_argument(
        '--tags',
        help='JSON string of experiment tags (e.g., \'{"version": "1.0", "type": "nda"}\')'
    )
    
    parser.add_argument(
        '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for generated files (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the pipeline."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(args.verbose)
        
        logger.info("Starting contract generation pipeline")
        logger.info(f"Skeleton: {args.skeleton}")
        logger.info(f"Data: {args.data}")
        
        # Validate skeleton file
        skeleton_path = validate_file_path(args.skeleton, MAX_SKELETON_FILE_SIZE, "Skeleton")
        
        # Create experiment folder structure
        exp_folder = create_experiment_folder(args.experiment, args.skeleton, args.data)
        
        # Load contract data
        logger.info("Loading contract data...")
        contract_data = load_contract_data(args.data)
        
        # Save extracted data to experiment folder
        with open(exp_folder / "reports" / "extracted_data.json", 'w') as f:
            # Create a serializable copy
            serializable_data = {k: v for k, v in contract_data.items() if k != '_metadata'}
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        # Display data extraction info
        if "_metadata" in contract_data:
            metadata = contract_data["_metadata"]
            logger.info(f"Data extracted using: {metadata.get('extraction_method', 'unknown')}")
            
            if metadata.get('source_type') == 'directory':
                logger.info(f"Processed {metadata.get('processed_files', 0)} files from directory")
            else:
                logger.info(f"Content length: {metadata.get('content_length', 0)} characters")
        
        logger.info(f"Loaded data for contract type: {contract_data.get('contract_type', 'unknown')}")
        logger.info(f"Experiment folder: {exp_folder}")
        
        # Process checklist
        checklist = None
        if args.checklist:
            checklist = [item.strip() for item in args.checklist.split(',') if item.strip()]
            logger.info(f"Using checklist with {len(checklist)} items")
        
        # Load reference contracts
        reference_contracts = None
        if args.references:
            logger.info("Loading reference contracts...")
            reference_contracts = load_reference_contracts(args.references)
            if reference_contracts:
                logger.info(f"Loaded {len(reference_contracts)} reference contracts")
        
        # Create output directory
        output_dir = create_output_directory(args.output_dir)
        
        # Generate output filename if not provided
        output_path = args.output
        if not output_path:
            contract_type = contract_data.get('contract_type', 'contract')
            output_path = exp_folder / "outputs" / f"{contract_type}_generated.docx"
        else:
            output_path = Path(output_path)
        
        # Parse experiment tags
        experiment_tags = None
        if args.tags:
            try:
                experiment_tags = json.loads(args.tags)
                if not isinstance(experiment_tags, dict):
                    raise ValueError("Tags must be a JSON object")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format in tags: {e}")
                return 1
        
        # Initialize pipeline
        enable_mlflow = not args.no_mlflow
        logger.info(f"MLflow tracking: {'enabled' if enable_mlflow else 'disabled'}")
        
        pipeline = QualityAssurancePipeline(enable_mlflow=enable_mlflow)
        
        # Run pipeline
        logger.info("Running quality assurance pipeline...")
        result = pipeline.process_contract(
            skeleton_path=str(skeleton_path),
            contract_data=contract_data,
            checklist=checklist,
            reference_contracts=reference_contracts,
            output_path=str(output_path),
            experiment_name=args.experiment,
            experiment_tags=experiment_tags,
            experiment_folder=exp_folder  # Pass experiment folder for temp files
        )
        
        # Save results to experiment folder
        results_summary = {
            "success": result.success,
            "iterations": result.iterations,
            "total_time": result.total_time,
            "output_file": str(output_path),
            "quality_scores": result.quality_scores or {},
            "quality_gates": [
                {
                    "gate_name": gate.gate_name,
                    "score": gate.score,
                    "threshold": gate.threshold,
                    "status": gate.status.value if hasattr(gate.status, 'value') else str(gate.status)
                } for gate in (result.quality_gates or [])
            ],
            "warnings": result.warnings or [],
            "experiment_folder": str(exp_folder)
        }
        
        with open(exp_folder / "reports" / "pipeline_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Copy final output to experiment folder if it exists elsewhere
        if output_path.exists() and not str(output_path).startswith(str(exp_folder)):
            import shutil
            final_output = exp_folder / "outputs" / output_path.name
            shutil.copy2(output_path, final_output)
            results_summary["output_file"] = str(final_output)
        
        # Display results
        print("\n" + "="*80)
        print("🎉 PIPELINE EXECUTION COMPLETED")
        print("="*80)
        
        print(f"\n📊 Generation Results:")
        print(f"• Success: {'✅ Yes' if result.success else '❌ No'}")
        print(f"• Iterations: {result.iterations}")
        print(f"• Total Time: {result.total_time:.2f} seconds")
        print(f"• Output File: {output_path}")
        print(f"• Experiment Folder: {exp_folder}")
        
        if result.quality_scores:
            print(f"\n📈 Quality Scores:")
            for metric, score in result.quality_scores.items():
                print(f"• {metric.upper()}: {score:.3f}")
        
        if result.quality_gates:
            print(f"\n🚪 Quality Gates:")
            passed_gates = sum(1 for gate in result.quality_gates if gate.status.value == "passed")
            total_gates = len(result.quality_gates)
            print(f"• Passed: {passed_gates}/{total_gates}")
            
            for gate in result.quality_gates:
                status_icon = "✅" if gate.status.value == "passed" else "❌"
                print(f"  {status_icon} {gate.gate_name}: {gate.score:.3f} (threshold: {gate.threshold})")
        
        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"• {warning}")
        
        if enable_mlflow and 'mlflow_run_id' in result.metadata:
            print(f"\n🔬 MLflow Tracking:")
            print(f"• Experiment: {args.experiment}")
            print(f"• Run ID: {result.metadata.get('mlflow_run_id', 'N/A')}")
        
        print(f"\n📁 Experiment Structure:")
        print(f"• Inputs: {exp_folder}/inputs/")
        print(f"• Outputs: {exp_folder}/outputs/")
        print(f"• Reports: {exp_folder}/reports/")
        print(f"• Logs: {exp_folder}/logs/")
        
        print(f"\n🎯 Overall Success: {'✅' if result.success else '❌'}")
        
        return 0 if result.success else 1
        
    except ProcessingError as e:
        logger.error(f"Processing error: {e}")
        if hasattr(e, 'suggestions') and e.suggestions:
            print(f"\n💡 Suggestions:")
            for suggestion in e.suggestions:
                print(f"• {suggestion}")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        error_handler.handle_error(e, {"stage": "pipeline_script"})
        return 1


if __name__ == "__main__":
    import time
    exit_code = main()
    sys.exit(exit_code)