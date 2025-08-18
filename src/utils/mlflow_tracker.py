"""MLflow integration for experiment tracking and model management."""

import logging
import os
import hashlib
import shutil
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

import mlflow
import mlflow.tracking
from mlflow.entities import Experiment, Run

from config.settings import settings

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow tracker for contract generation experiments."""
    
    def __init__(self):
        self.tracking_uri = settings.mlflow_tracking_uri
        self.experiment_name = settings.mlflow_experiment_name
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except Exception:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow initialized with experiment: {self.experiment_name}")
    
    def start_contract_generation_run(self, contract_context: Dict[str, Any],
                                    run_name: Optional[str] = None) -> str:
        """Start a new MLflow run for contract generation.
        
        Args:
            contract_context: Context information about the contract
            run_name: Optional name for the run
            
        Returns:
            Run ID of the started run
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            contract_type = contract_context.get("contract_type", "unknown")
            run_name = f"contract_gen_{contract_type}_{timestamp}"
        
        run = mlflow.start_run(run_name=run_name)
        
        # Log contract context as parameters
        self._log_contract_context(contract_context)
        
        # Log system information
        mlflow.log_param("openai_model", settings.openai_model)
        mlflow.log_param("temperature", settings.openai_temperature)
        mlflow.log_param("quality_gates_enabled", settings.quality_gate_enabled)
        mlflow.log_param("max_regeneration_attempts", settings.max_regeneration_attempts)
        
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id
    
    def log_generation_metrics(self, quality_scores: Dict[str, float],
                             generation_metadata: Dict[str, Any]):
        """Log generation quality metrics and metadata.
        
        Args:
            quality_scores: Dictionary of quality metric scores
            generation_metadata: Metadata about the generation process
        """
        # Log quality metrics
        for metric_name, score in quality_scores.items():
            mlflow.log_metric(f"quality_{metric_name}", score)
        
        # Log generation metadata
        if "generation_time" in generation_metadata:
            mlflow.log_metric("generation_time_seconds", generation_metadata["generation_time"])
        
        if "iterations" in generation_metadata:
            mlflow.log_metric("quality_iterations", generation_metadata["iterations"])
        
        if "prompt_tokens" in generation_metadata:
            mlflow.log_metric("prompt_tokens", generation_metadata["prompt_tokens"])
        
        if "completion_tokens" in generation_metadata:
            mlflow.log_metric("completion_tokens", generation_metadata["completion_tokens"])
        
        # Log overall quality score
        if quality_scores:
            overall_score = self._calculate_overall_score(quality_scores)
            mlflow.log_metric("overall_quality_score", overall_score)
        
        logger.info("Logged generation metrics to MLflow")
    
    def log_quality_gates(self, quality_gates: List[Any]):
        """Log quality gate results.
        
        Args:
            quality_gates: List of quality gate results (QualityGateResult objects or dicts)
        """
        gates_passed = 0
        total_gates = len(quality_gates)
        
        for gate in quality_gates:
            # Handle both QualityGateResult objects and dictionaries
            if hasattr(gate, 'gate_name'):
                # QualityGateResult object
                gate_name = gate.gate_name
                status = gate.status.value if hasattr(gate.status, 'value') else str(gate.status)
                score = gate.score
                threshold = gate.threshold
            else:
                # Dictionary
                gate_name = gate.get("gate_name", "unknown")
                status = gate.get("status", "unknown")
                score = gate.get("score", 0.0)
                threshold = gate.get("threshold", 0.0)
            
            # Log individual gate results
            mlflow.log_metric(f"gate_{gate_name}_score", score)
            mlflow.log_metric(f"gate_{gate_name}_threshold", threshold)
            mlflow.log_param(f"gate_{gate_name}_status", status)
            
            if status == "passed":
                gates_passed += 1
        
        # Log summary metrics
        if total_gates > 0:
            pass_rate = gates_passed / total_gates
            mlflow.log_metric("quality_gates_pass_rate", pass_rate)
            mlflow.log_metric("quality_gates_passed", gates_passed)
            mlflow.log_metric("quality_gates_total", total_gates)
        
        logger.info(f"Logged {total_gates} quality gates to MLflow")
    
    def log_contract_artifacts(self, contract_path: str, skeleton_path: str,
                             quality_report: Optional[Dict[str, Any]] = None):
        """Log contract files and reports as artifacts.
        
        Args:
            contract_path: Path to the generated contract
            skeleton_path: Path to the skeleton file
            quality_report: Optional quality assessment report
        """
        # Log generated contract
        if os.path.exists(contract_path):
            mlflow.log_artifact(contract_path, "contracts")
        
        # Log skeleton
        if os.path.exists(skeleton_path):
            mlflow.log_artifact(skeleton_path, "skeletons")
        
        # Log quality report
        if quality_report:
            report_path = "temp_quality_report.json"
            with open(report_path, 'w') as f:
                json.dump(quality_report, f, indent=2)
            mlflow.log_artifact(report_path, "reports")
            os.remove(report_path)  # Clean up temp file
        
        logger.info("Logged contract artifacts to MLflow")
    
    def log_code_artifacts(self, additional_files: Optional[List[str]] = None):
        """Log code files and version information for reproducibility.
        
        Args:
            additional_files: Optional list of additional files to log
        """
        # Core source files that affect contract generation
        core_files = [
            "src/core/content_generator.py",
            "src/core/quality_pipeline.py", 
            "src/core/document_processor.py",
            "src/evaluation/metrics.py",
            "src/evaluation/llm_judge.py",
            "config/settings.py"
        ]
        
        # Add any additional files
        if additional_files:
            core_files.extend(additional_files)
        
        # Log individual source files
        for file_path in core_files:
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path, "source_code")
        
        # Create and log code snapshot
        self._create_code_snapshot()
        
        # Log git information if available
        self._log_git_info()
        
        # Log environment information
        self._log_environment_info()
        
        logger.info("Logged code artifacts and version information")
    
    def _create_code_snapshot(self):
        """Create a snapshot of the entire codebase."""
        snapshot_dir = "temp_code_snapshot"
        
        try:
            # Create temporary directory
            os.makedirs(snapshot_dir, exist_ok=True)
            
            # Copy core directories
            dirs_to_copy = ["src", "config", "examples"]
            for dir_name in dirs_to_copy:
                if os.path.exists(dir_name):
                    shutil.copytree(
                        dir_name, 
                        os.path.join(snapshot_dir, dir_name),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'),
                        dirs_exist_ok=True
                    )
            
            # Copy key files
            key_files = ["requirements.txt", "README.md", "PRD_Lawyer_Contract_Creation.md"]
            for file_name in key_files:
                if os.path.exists(file_name):
                    shutil.copy2(file_name, snapshot_dir)
            
            # Create code manifest
            self._create_code_manifest(snapshot_dir)
            
            # Log the entire snapshot as a compressed artifact
            mlflow.log_artifacts(snapshot_dir, "code_snapshot")
            
        finally:
            # Clean up temporary directory
            if os.path.exists(snapshot_dir):
                shutil.rmtree(snapshot_dir)
    
    def _create_code_manifest(self, snapshot_dir: str):
        """Create a manifest file with code hashes and metadata."""
        manifest = {
            "creation_time": datetime.now().isoformat(),
            "files": {},
            "total_files": 0,
            "total_size_bytes": 0
        }
        
        # Walk through the snapshot directory
        for root, dirs, files in os.walk(snapshot_dir):
            for file in files:
                if file.endswith(('.py', '.md', '.txt', '.json', '.yaml', '.yml')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, snapshot_dir)
                    
                    # Calculate file hash
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    # Get file stats
                    file_stats = os.stat(file_path)
                    
                    manifest["files"][relative_path] = {
                        "hash": file_hash,
                        "size_bytes": file_stats.st_size,
                        "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    }
                    manifest["total_files"] += 1
                    manifest["total_size_bytes"] += file_stats.st_size
        
        # Save manifest
        manifest_path = os.path.join(snapshot_dir, "code_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Log key manifest info as parameters
        mlflow.log_param("code_files_count", manifest["total_files"])
        mlflow.log_param("code_size_bytes", manifest["total_size_bytes"])
        mlflow.log_param("code_snapshot_time", manifest["creation_time"])
    
    def _log_git_info(self):
        """Log git repository information if available."""
        try:
            # Get current commit hash
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            mlflow.log_param("git_commit_hash", commit_hash)
            
            # Get current branch
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            mlflow.log_param("git_branch", branch)
            
            # Check if there are uncommitted changes
            try:
                subprocess.check_output(
                    ['git', 'diff', '--quiet'],
                    stderr=subprocess.DEVNULL
                )
                subprocess.check_output(
                    ['git', 'diff', '--cached', '--quiet'],
                    stderr=subprocess.DEVNULL
                )
                mlflow.log_param("git_clean", True)
            except subprocess.CalledProcessError:
                mlflow.log_param("git_clean", False)
                mlflow.set_tag("warning", "Uncommitted changes present")
            
            # Get commit message
            commit_msg = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%B'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            mlflow.log_param("git_commit_message", commit_msg[:100])  # Truncate if too long
            
            # Get author info
            author = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%an <%ae>'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            mlflow.log_param("git_author", author)
            
            logger.info(f"Logged git info: {commit_hash[:8]} on {branch}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            mlflow.log_param("git_available", False)
            logger.warning("Git not available or not a git repository")
    
    def _log_environment_info(self):
        """Log environment and dependency information."""
        import sys
        import platform
        
        # System information
        mlflow.log_param("python_version", sys.version.split()[0])
        mlflow.log_param("platform", platform.platform())
        mlflow.log_param("architecture", platform.architecture()[0])
        
        # Try to get package versions for key dependencies
        key_packages = [
            "openai", "python-docx", "nltk", "rouge-score", 
            "scikit-learn", "unbabel-comet", "mlflow"
        ]
        
        for package in key_packages:
            try:
                import importlib.metadata
                version = importlib.metadata.version(package)
                mlflow.log_param(f"package_{package.replace('-', '_')}_version", version)
            except importlib.metadata.PackageNotFoundError:
                mlflow.log_param(f"package_{package.replace('-', '_')}_version", "not_installed")
        
        # Log requirements.txt if available
        if os.path.exists("requirements.txt"):
            mlflow.log_artifact("requirements.txt", "environment")
        
        logger.info("Logged environment information")
    
    def log_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """Log detailed evaluation results.
        
        Args:
            evaluation_results: Dictionary containing evaluation details
        """
        for metric_name, result in evaluation_results.items():
            if hasattr(result, 'score'):
                mlflow.log_metric(f"eval_{metric_name}_score", result.score)
                mlflow.log_param(f"eval_{metric_name}_threshold", result.threshold)
                mlflow.log_param(f"eval_{metric_name}_passed", result.passed_threshold)
            
            # Log detailed results as JSON
            if hasattr(result, 'details'):
                details_path = f"temp_{metric_name}_details.json"
                with open(details_path, 'w') as f:
                    json.dump(result.details, f, indent=2, default=str)
                mlflow.log_artifact(details_path, "evaluation_details")
                os.remove(details_path)
        
        logger.info("Logged evaluation results to MLflow")
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run.
        
        Args:
            status: Status of the run (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run with status: {status}")
    
    def log_error(self, error_message: str, error_type: str = "GenerationError"):
        """Log error information to MLflow.
        
        Args:
            error_message: Error message
            error_type: Type of error
        """
        mlflow.log_param("error_type", error_type)
        mlflow.log_param("error_message", error_message)
        mlflow.set_tag("status", "FAILED")
        
        logger.error(f"Logged error to MLflow: {error_type} - {error_message}")
    
    def _log_contract_context(self, context: Dict[str, Any]):
        """Log contract context as parameters."""
        # Log basic contract information
        mlflow.log_param("contract_type", context.get("contract_type", "unknown"))
        mlflow.log_param("client_name", context.get("client_name", "unknown"))
        mlflow.log_param("provider_name", context.get("provider_name", "unknown"))
        
        # Log contract value if available
        if "contract_value" in context:
            mlflow.log_param("contract_value", str(context["contract_value"]))
        
        # Log service description length
        if "service_description" in context:
            desc_length = len(str(context["service_description"]))
            mlflow.log_metric("service_description_length", desc_length)
        
        # Log checklist size if available
        if "checklist" in context and isinstance(context["checklist"], list):
            mlflow.log_metric("checklist_items", len(context["checklist"]))
        
        # Log contract data complexity
        data_complexity = len(context)
        mlflow.log_metric("contract_data_fields", data_complexity)
    
    def _calculate_overall_score(self, quality_scores: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        if not quality_scores:
            return 0.0
        
        # Weighted average based on metric importance
        weights = {
            "bleu": 0.15,
            "rouge": 0.15,
            "meteor": 0.15,
            "comet": 0.15,
            "redundancy": 0.1,  # Lower is better for redundancy
            "completeness": 0.15,
            "llm_judge": 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in quality_scores.items():
            if metric in weights:
                weight = weights[metric]
                # For redundancy, invert the score (lower is better)
                if metric == "redundancy":
                    score = 1.0 - score
                weighted_sum += score * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_experiment_runs(self, max_results: int = 100) -> List[Run]:
        """Get runs from the current experiment.
        
        Args:
            max_results: Maximum number of runs to return
            
        Returns:
            List of MLflow Run objects
        """
        return mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )
    
    def get_best_runs_by_metric(self, metric_name: str, top_k: int = 5) -> List[Run]:
        """Get top runs by a specific metric.
        
        Args:
            metric_name: Name of the metric to sort by
            top_k: Number of top runs to return
            
        Returns:
            List of top MLflow Run objects
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=top_k
        )
        return runs
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Dictionary containing comparison results
        """
        comparison_data = {
            "runs": [],
            "metrics_comparison": {},
            "parameters_comparison": {}
        }
        
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            comparison_data["runs"].append({
                "run_id": run_id,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "status": run.info.status,
                "metrics": run.data.metrics,
                "params": run.data.params
            })
        
        # Aggregate metrics for comparison
        if comparison_data["runs"]:
            all_metrics = set()
            for run_data in comparison_data["runs"]:
                all_metrics.update(run_data["metrics"].keys())
            
            for metric in all_metrics:
                comparison_data["metrics_comparison"][metric] = [
                    run_data["metrics"].get(metric, None) 
                    for run_data in comparison_data["runs"]
                ]
        
        return comparison_data
    
    def log_model_performance(self, model_metrics: Dict[str, float]):
        """Log model performance metrics.
        
        Args:
            model_metrics: Dictionary of model performance metrics
        """
        for metric_name, value in model_metrics.items():
            mlflow.log_metric(f"model_{metric_name}", value)
        
        logger.info("Logged model performance metrics to MLflow")
    
    def register_model(self, model_path: str, model_name: str, 
                      model_version: Optional[str] = None):
        """Register a model in MLflow Model Registry.
        
        Args:
            model_path: Path to the model artifacts
            model_name: Name for the registered model
            model_version: Optional version identifier
        """
        try:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_path}"
            
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            logger.info(f"Registered model: {model_name}, version: {registered_model.version}")
            return registered_model
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            return None


def track_contract_generation(func):
    """Decorator for tracking contract generation functions with MLflow."""
    def wrapper(*args, **kwargs):
        tracker = MLflowTracker()
        
        # Extract context from function arguments
        context = kwargs.get("contract_data", {})
        
        try:
            # Start MLflow run
            run_id = tracker.start_contract_generation_run(context)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Log results if successful
            if hasattr(result, 'quality_scores'):
                tracker.log_generation_metrics(
                    result.quality_scores, 
                    result.metadata
                )
            
            if hasattr(result, 'quality_gates'):
                tracker.log_quality_gates([
                    {
                        "gate_name": gate.gate_name,
                        "status": gate.status.value,
                        "score": gate.score,
                        "threshold": gate.threshold
                    }
                    for gate in result.quality_gates
                ])
            
            tracker.end_run("FINISHED")
            return result
            
        except Exception as e:
            tracker.log_error(str(e))
            tracker.end_run("FAILED")
            raise
    
    return wrapper
