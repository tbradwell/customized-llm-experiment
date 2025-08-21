"""Simple pipeline for skeleton-based contract generation with MLflow tracking."""

import logging
import re
from pathlib import Path
from typing import Dict, Any

import mlflow
from docx import Document

from config.settings import settings
from .skeleton_fill_processor import SkeletonFillProcessor
from ..evaluation.metrics import MetricsCalculator
from ..utils.doc_handler import DocHandler
from ..utils.error_handler import ProcessingError

logger = logging.getLogger(__name__)


class SimplePipeline:
    """Simple pipeline for contract generation using skeleton replacement."""
    
    def __init__(self):
        self.skeleton_fill_processor = SkeletonFillProcessor()
        self.metrics_calc = MetricsCalculator()
    
    def _setup_mlflow(self, experiment_name: str = "Default"):
        """Setup MLflow tracking to store data in mlruns directory structure."""
        try:
            # Set MLflow tracking URI to standard mlruns directory
            tracking_uri = "./mlruns"
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
            
            # Use the default experiment (experiment ID = 0)
            # This creates the standard mlruns/0/{run_id}/ structure
            mlflow.set_experiment(experiment_name)
            logger.info("Using default MLflow experiment (ID: 0) for standard mlruns structure")
            
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
            # Fallback: just set tracking URI and let MLflow handle the rest
            mlflow.set_tracking_uri("./mlruns")
    
    def run_pipeline(self, new_data_dir: str, skeleton_path: str, output_dir: str,
                    contract_type: str = "legal_claim", test_type: str = "amit_example",
                    experiment_name: str = "Default") -> Dict[str, Any]:
        """Run the complete pipeline with MLflow tracking.
        
        Args:
            new_data_dir: Directory containing new data files
            skeleton_path: Path to skeleton document
            output_dir: Directory to save outputs
            contract_type: Type of contract being generated
            test_type: Type of test for MLflow tagging
            
        Returns:
            Dictionary with pipeline results
        """
        # Setup MLflow
        self._setup_mlflow(experiment_name=experiment_name)
        
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            try:
                # Set MLflow tags and parameters
                self._setup_mlflow_run(test_type, contract_type)
                
                # Step 1: Load new data
                logger.info("Step 1: Loading new data")
                new_data = self._load_new_data(new_data_dir)
                mlflow.log_param("new_data_length", len(new_data))
                
                # Step 2: Load skeleton
                logger.info("Step 2: Loading skeleton document")
                skeleton_doc = Document(skeleton_path)
                
                # Count initial placeholders
                initial_text = DocHandler.extract_text_from_doc(skeleton_doc)
                initial_placeholders = len(re.findall(r'\{[^}]*\}', initial_text))
                mlflow.log_metric("initial_placeholders", initial_placeholders)
                
                # Step 3: Fill skeleton with new data
                logger.info("Step 3: Filling skeleton with new data")
                processed_doc, iterations = self.skeleton_fill_processor.fill_skeleton_with_data(
                    skeleton_doc, new_data
                )
                mlflow.log_metric("completion_iterations", iterations)
                
                # Step 4: Verify completion
                final_text = DocHandler.extract_text_from_doc(processed_doc)
                final_placeholders = len(re.findall(r'\{[^}]*\}', final_text))
                mlflow.log_metric("final_placeholders", final_placeholders)
                
                # Set completion status
                completion_status = "COMPLETE" if final_placeholders == 0 else "INCOMPLETE"
                mlflow.log_param("completion_status", completion_status)
                
                # Step 5: Save outputs and log artifacts
                logger.info("Step 5: Saving outputs and logging artifacts")
                artifacts_info = self._save_outputs_and_artifacts(
                    processed_doc, skeleton_path, final_text, run_id, output_dir
                )
                
                # Step 6: Run evaluation
                logger.info("Step 6: Running evaluation")
                evaluation_results = self._run_evaluation(
                    final_text, output_dir, artifacts_info
                )
                
                # Log evaluation metrics
                if evaluation_results:
                    for metric_name, score in evaluation_results.items():
                        if isinstance(score, (int, float)):
                            mlflow.log_metric(f"quality_{metric_name}", score)
                
                # Set final status and return results
                mlflow.set_tag("status", "FINISHED")
                return self._create_results_summary(
                    run_id, initial_placeholders, final_placeholders, 
                    iterations, completion_status, final_text, 
                    evaluation_results, artifacts_info
                )
                
            except Exception as e:
                logger.error(f"Pipeline failed: {str(e)}")
                mlflow.set_tag("status", "FAILED")
                mlflow.log_param("error", str(e))
                raise ProcessingError(f"Pipeline failed: {str(e)}")
    
    def _setup_mlflow_run(self, test_type: str, contract_type: str):
        """Setup MLflow run tags and parameters."""
        mlflow.set_tag("approach", "simple_skeleton")
        mlflow.set_tag("test_type", test_type)
        mlflow.set_tag("status", "RUNNING")
        mlflow.log_param("contract_type", contract_type)
        mlflow.log_param("openai_model", settings.openai_model)
    
    def _load_new_data(self, new_data_dir: str) -> str:
        """Load and combine new data from directory."""
        from ..processors.data_loader import DataLoader
        
        data_loader = DataLoader()
        return data_loader.load_new_data(new_data_dir)
    
    def _save_outputs_and_artifacts(self, processed_doc: Document, skeleton_path: str,
                                   final_text: str, run_id: str, output_dir: str) -> Dict[str, str]:
        """Save outputs and log MLflow artifacts."""
        from ..utils.artifact_manager import ArtifactManager
        
        artifact_manager = ArtifactManager()
        return artifact_manager.save_outputs_and_artifacts(
            processed_doc, skeleton_path, final_text, run_id, output_dir
        )
    
    def _run_evaluation(self, final_text: str, output_dir: str, artifacts_info: Dict[str, str]) -> Dict[str, float]:
        """Run evaluation metrics against ground truth if available."""
        from ..evaluation.contract_evaluator import ContractEvaluator
        
        evaluator = ContractEvaluator()
        return evaluator.run_evaluation(final_text, output_dir, artifacts_info)
    
    def _create_results_summary(self, run_id: str, initial_placeholders: int, 
                               final_placeholders: int, iterations: int, 
                               completion_status: str, final_text: str,
                               evaluation_results: Dict, artifacts_info: Dict) -> Dict[str, Any]:
        """Create results summary dictionary."""
        return {
            "run_id": run_id,
            "success": True,
            "initial_placeholders": initial_placeholders,
            "final_placeholders": final_placeholders,
            "completion_iterations": iterations,
            "completion_status": completion_status,
            "final_content_length": len(final_text),
            "evaluation_results": evaluation_results,
            "artifacts": artifacts_info
        }
