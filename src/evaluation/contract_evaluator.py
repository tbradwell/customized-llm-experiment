"""Contract evaluation and metrics calculation for quality assessment."""

import logging
from pathlib import Path
from typing import Dict

from .metrics import MetricsCalculator
from ..utils.doc_handler import DocHandler

logger = logging.getLogger(__name__)


class ContractEvaluator:
    """Evaluates contracts and calculates quality metrics."""
    
    def __init__(self):
        self.metrics_calc = MetricsCalculator()
    
    def run_evaluation(self, final_text: str, output_dir: str, artifacts_info: Dict[str, str]) -> Dict[str, float]:
        """Run evaluation metrics against ground truth if available."""
        
        # Look for ground truth file
        gt_path = Path(output_dir).parent / "gt.docx"
        if not gt_path.exists():
            logger.info("No ground truth file found, skipping evaluation")
            return {}
        
        try:
            # Extract ground truth text
            gt_text = DocHandler.extract_text_from_docx(str(gt_path))
            
            # Calculate metrics
            metrics = {}
            reference_texts = [gt_text]
            
            # BLEU Score
            try:
                bleu_result = self.metrics_calc.calculate_bleu_score(final_text, reference_texts)
                metrics['bleu'] = bleu_result.score
            except Exception as e:
                logger.warning(f"BLEU calculation failed: {e}")
                metrics['bleu'] = 0.0
            
            # ROUGE Scores
            try:
                rouge_result = self.metrics_calc.calculate_rouge_scores(final_text, reference_texts)
                metrics['rouge'] = rouge_result.details.get('rougeL_f', 0.0)
            except Exception as e:
                logger.warning(f"ROUGE calculation failed: {e}")
                metrics['rouge'] = 0.0
            
            # METEOR Score
            try:
                meteor_result = self.metrics_calc.calculate_meteor_score(final_text, reference_texts)
                metrics['meteor'] = meteor_result.score
            except Exception as e:
                logger.warning(f"METEOR calculation failed: {e}")
                metrics['meteor'] = 0.0
            
            # COMET Score
            try:
                comet_result = self.metrics_calc.calculate_comet_score(final_text, reference_texts)
                metrics['comet'] = comet_result.score
            except Exception as e:
                logger.warning(f"COMET calculation failed: {e}")
                metrics['comet'] = 0.0
            
            # Redundancy Score
            try:
                redundancy_result = self.metrics_calc.calculate_redundancy_score(final_text)
                metrics['redundancy'] = redundancy_result.score
            except Exception as e:
                logger.warning(f"Redundancy calculation failed: {e}")
                metrics['redundancy'] = 0.0
            
            # Completeness Score
            try:
                # Define standard legal contract elements for completeness check
                required_elements = [
                    "parties", "party", "client", "provider", "contractor",
                    "consideration", "payment", "fee", "amount", "compensation",
                    "scope", "services", "work", "obligations", "duties",
                    "term", "duration", "period", "effective date", "expiration",
                    "termination", "breach", "default", "cancellation",
                    "liability", "damages", "indemnification", "responsibility",
                    "governing law", "jurisdiction", "dispute resolution",
                    "signatures", "execution", "agreement", "contract"
                ]
                completeness_result = self.metrics_calc.calculate_completeness_score(
                    final_text, required_elements
                )
                metrics['completeness'] = completeness_result.score
            except Exception as e:
                logger.warning(f"Completeness calculation failed: {e}")
                metrics['completeness'] = 0.0
            
            # LLM Judge Score  
            try:
                from .llm_judge import LLMJudge
                llm_judge = LLMJudge()
                llm_result = llm_judge.evaluate_contract(final_text, {"contract_type": "legal_claim"}, reference_texts)
                metrics['llm_judge'] = llm_result.score
            except Exception as e:
                logger.warning(f"LLM Judge calculation failed: {e}")
                metrics['llm_judge'] = 0.0
            
            # Print all metrics at the end
            logger.info("📊 ALL EVALUATION METRICS COMPLETED:")
            for metric_name, score in metrics.items():
                logger.info(f"  • {metric_name.upper()}: {score:.3f}")
            
            logger.info(f"Evaluation completed with {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
