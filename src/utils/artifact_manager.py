"""Artifact management utilities for contract generation."""

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Any

import mlflow
from docx import Document

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Handles saving outputs and MLflow artifacts."""
    
    def save_outputs_and_artifacts(self, processed_doc: Document, skeleton_path: str,
                                   final_text: str, run_id: str, output_dir: str) -> Dict[str, str]:
        """Save outputs and log MLflow artifacts."""
        
        # Create output directory structure
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save final contract
        contract_path = output_path / "contract_simple.docx"
        processed_doc.save(contract_path)
        
        # Copy original skeleton for reference
        skeleton_copy_path = output_path / "sekeleton_oracle.docx"
        shutil.copy2(skeleton_path, skeleton_copy_path)
        
        # Create quality assessment report
        quality_report = self._create_quality_report(final_text)
        quality_report_path = output_path / "quality_assessment.json"
        
        with open(quality_report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(str(contract_path), "contracts")
        mlflow.log_artifact(str(skeleton_copy_path), "skeletons")
        mlflow.log_artifact(str(quality_report_path), "reports")
        
        return {
            "contract_path": str(contract_path),
            "skeleton_path": str(skeleton_copy_path),
            "quality_report_path": str(quality_report_path)
        }
    
    def _create_quality_report(self, final_text: str) -> Dict[str, Any]:
        """Create quality assessment report."""
        
        return {
            "final_content_length": len(final_text),
            "placeholder_analysis": {
                "remaining_placeholders": len(re.findall(r'\{[^}]*\}', final_text)),
                "placeholder_examples": re.findall(r'\{[^}]*\}', final_text)[:5]
            },
            "content_analysis": {
                "has_real_names": any(name in final_text for name in ["זואי", "מור", "גיל"]),
                "has_real_address": "חיים הרצוג" in final_text,
                "has_real_phone": "0542477683" in final_text,
                "unwanted_content": {
                    "doc_references": "DOC-" in final_text,
                    "generic_placeholders": "יינתן במועד מאוחר יותר" in final_text
                }
            },
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
