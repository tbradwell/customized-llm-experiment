"""Tests for evaluation metrics functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.evaluation.metrics import (
    MetricsCalculator, COMETEvaluator, EvaluationResult
)


class TestMetricsCalculator:
    """Test suite for MetricsCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()
    
    def test_calculate_bleu_score_basic(self):
        """Test basic BLEU score calculation."""
        generated_text = "The client agrees to pay the provider for services rendered."
        reference_texts = [
            "The client agrees to pay the provider for all services rendered.",
            "Client shall pay provider for services provided."
        ]
        
        result = self.calculator.calculate_bleu_score(generated_text, reference_texts)
        
        assert isinstance(result, EvaluationResult)
        assert result.metric_name == "BLEU"
        assert 0.0 <= result.score <= 1.0
        assert "generated_token_count" in result.details
        assert "reference_count" in result.details
    
    def test_calculate_bleu_score_empty_input(self):
        """Test BLEU score with empty input."""
        result = self.calculator.calculate_bleu_score("", ["reference text"])
        
        assert result.metric_name == "BLEU"
        assert result.score >= 0.0
        assert not result.passed_threshold
    
    def test_calculate_rouge_scores_basic(self):
        """Test basic ROUGE scores calculation."""
        generated_text = "The contract specifies payment terms and delivery schedule."
        reference_texts = [
            "The agreement outlines payment terms and delivery requirements.",
            "Contract includes payment conditions and schedule details."
        ]
        
        result = self.calculator.calculate_rouge_scores(generated_text, reference_texts)
        
        assert isinstance(result, EvaluationResult)
        assert result.metric_name == "ROUGE"
        assert 0.0 <= result.score <= 1.0
        assert "rouge1_avg" in result.details
        assert "rouge2_avg" in result.details
        assert "rougeL_avg" in result.details
    
    def test_calculate_meteor_score_basic(self):
        """Test basic METEOR score calculation."""
        generated_text = "The provider will deliver services as specified."
        reference_texts = [
            "The provider shall deliver services as outlined.",
            "Provider must deliver specified services."
        ]
        
        result = self.calculator.calculate_meteor_score(generated_text, reference_texts)
        
        assert isinstance(result, EvaluationResult)
        assert result.metric_name == "METEOR"
        assert 0.0 <= result.score <= 1.0
        assert "individual_scores" in result.details
        assert "reference_count" in result.details
    
    def test_calculate_redundancy_score_basic(self):
        """Test redundancy score calculation."""
        # Text with some redundancy
        redundant_text = """
        The client agrees to pay. The client agrees to pay promptly.
        Payment shall be made by client. Client must make payment.
        Services will be provided. The provider will provide services.
        """
        
        result = self.calculator.calculate_redundancy_score(redundant_text)
        
        assert isinstance(result, EvaluationResult)
        assert result.metric_name == "Redundancy"
        assert 0.0 <= result.score <= 1.0
        assert "sentence_count" in result.details
        assert "avg_similarity" in result.details
    
    def test_calculate_redundancy_score_minimal_text(self):
        """Test redundancy score with minimal text."""
        minimal_text = "Single sentence."
        
        result = self.calculator.calculate_redundancy_score(minimal_text)
        
        assert result.metric_name == "Redundancy"
        assert result.score == 0.0  # No redundancy in single sentence
        assert result.passed_threshold is True
    
    def test_calculate_completeness_score_basic(self):
        """Test completeness score calculation."""
        contract_text = """
        This agreement includes confidentiality provisions,
        payment terms, delivery schedule, and termination conditions.
        The parties agree to binding arbitration for disputes.
        """
        
        required_elements = [
            "confidentiality",
            "payment terms", 
            "delivery schedule",
            "termination",
            "arbitration"
        ]
        
        result = self.calculator.calculate_completeness_score(contract_text, required_elements)
        
        assert isinstance(result, EvaluationResult)
        assert result.metric_name == "Completeness"
        assert 0.0 <= result.score <= 1.0
        assert "total_required" in result.details
        assert "found_count" in result.details
        assert "missing_count" in result.details
        assert result.details["total_required"] == len(required_elements)
    
    def test_calculate_completeness_score_no_requirements(self):
        """Test completeness score with no requirements."""
        result = self.calculator.calculate_completeness_score("Any text", [])
        
        assert result.score == 1.0  # Complete if no requirements
        assert result.passed_threshold is True
    
    def test_calculate_completeness_score_missing_elements(self):
        """Test completeness score with missing elements."""
        contract_text = "This contract has payment terms."
        required_elements = ["payment terms", "confidentiality", "termination"]
        
        result = self.calculator.calculate_completeness_score(contract_text, required_elements)
        
        assert result.score < 1.0  # Not complete
        assert len(result.details["missing_elements"]) > 0
        assert "confidentiality" in result.details["missing_elements"]
        assert "termination" in result.details["missing_elements"]


class TestCOMETEvaluator:
    """Test suite for COMETEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = COMETEvaluator()
    
    def test_comet_evaluator_initialization(self):
        """Test COMET evaluator initialization."""
        # Should initialize without errors
        assert self.evaluator is not None
    
    @patch('src.evaluation.metrics.COMETEvaluator._calculate_mock_comet_score')
    def test_calculate_comet_score_fallback(self, mock_comet):
        """Test COMET score calculation fallback."""
        mock_comet.return_value = 0.8
        
        generated_text = "High quality contract text with proper legal language."
        reference_texts = ["Reference contract with similar quality and structure."]
        
        result = self.evaluator.calculate_comet_score(generated_text, reference_texts)
        
        assert isinstance(result, EvaluationResult)
        assert result.metric_name == "COMET"
        assert 0.0 <= result.score <= 1.0
        assert "reference_count" in result.details
    
    def test_calculate_mock_comet_score(self):
        """Test mock COMET score calculation."""
        generated_text = "Contract text for evaluation."
        reference_texts = ["Similar contract text for comparison."]
        
        score = self.evaluator._calculate_mock_comet_score(generated_text, reference_texts)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_calculate_mock_comet_score_no_references(self):
        """Test mock COMET score with no references."""
        score = self.evaluator._calculate_mock_comet_score("Text", [])
        assert score == 0.5  # Default score


class TestEvaluationResult:
    """Test suite for EvaluationResult dataclass."""
    
    def test_evaluation_result_creation(self):
        """Test EvaluationResult creation."""
        result = EvaluationResult(
            metric_name="TEST",
            score=0.85,
            details={"test": "value"},
            passed_threshold=True,
            threshold=0.8
        )
        
        assert result.metric_name == "TEST"
        assert result.score == 0.85
        assert result.passed_threshold is True
        assert result.threshold == 0.8
        assert result.details["test"] == "value"


class TestMetricsIntegration:
    """Integration tests for metrics calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()
        self.comet_evaluator = COMETEvaluator()
    
    def test_full_evaluation_suite(self):
        """Test running full evaluation suite on contract text."""
        generated_contract = """
        SERVICE AGREEMENT
        
        This Service Agreement is entered into between Acme Corp and Provider LLC.
        
        1. SCOPE OF SERVICES
        Provider agrees to deliver consulting services including strategy development,
        implementation planning, and ongoing support.
        
        2. PAYMENT TERMS
        Total contract value is $75,000 payable in quarterly installments.
        Payment terms are Net 30 days from invoice date.
        
        3. CONFIDENTIALITY
        Both parties agree to maintain confidentiality of proprietary information.
        
        4. TERMINATION
        Either party may terminate with 60 days written notice.
        """
        
        reference_contracts = [
            """
            PROFESSIONAL SERVICES AGREEMENT
            
            This agreement is between Client Company and Service Provider.
            
            Services include strategic consulting and implementation support.
            Contract value is $80,000 with quarterly payment schedule.
            Confidentiality provisions protect proprietary information.
            Termination requires 30 days notice.
            """
        ]
        
        # Run all evaluation metrics
        bleu_result = self.calculator.calculate_bleu_score(
            generated_contract, reference_contracts, threshold=0.3
        )
        
        rouge_result = self.calculator.calculate_rouge_scores(
            generated_contract, reference_contracts, threshold=0.3
        )
        
        meteor_result = self.calculator.calculate_meteor_score(
            generated_contract, reference_contracts, threshold=0.3
        )
        
        redundancy_result = self.calculator.calculate_redundancy_score(
            generated_contract, threshold=0.3
        )
        
        completeness_result = self.calculator.calculate_completeness_score(
            generated_contract, 
            ["services", "payment", "confidentiality", "termination"],
            threshold=0.8
        )
        
        comet_result = self.comet_evaluator.calculate_comet_score(
            generated_contract, reference_contracts, threshold=0.3
        )
        
        # Verify all results
        results = [bleu_result, rouge_result, meteor_result, 
                  redundancy_result, completeness_result, comet_result]
        
        for result in results:
            assert isinstance(result, EvaluationResult)
            assert 0.0 <= result.score <= 1.0
            assert isinstance(result.details, dict)
            assert isinstance(result.passed_threshold, bool)
        
        # Specific checks
        assert completeness_result.score >= 0.8  # Should find most required elements
        assert redundancy_result.score <= 0.5   # Should have reasonable redundancy
    
    def test_evaluation_with_poor_quality_text(self):
        """Test evaluation with poor quality contract text."""
        poor_contract = "Bad contract. Very bad. Not good at all. Bad bad bad."
        reference_contracts = ["High quality professional contract with proper legal language."]
        
        bleu_result = self.calculator.calculate_bleu_score(poor_contract, reference_contracts)
        rouge_result = self.calculator.calculate_rouge_scores(poor_contract, reference_contracts)
        redundancy_result = self.calculator.calculate_redundancy_score(poor_contract)
        
        # Poor quality should result in lower scores
        assert bleu_result.score < 0.5
        assert rouge_result.score < 0.5
        # High redundancy due to repeated "bad"
        assert redundancy_result.score > 0.3
    
    def test_evaluation_error_handling(self):
        """Test evaluation error handling."""
        # Test with problematic input
        problematic_inputs = ["", None, "x" * 10000]  # Empty, None, very long
        
        for input_text in problematic_inputs:
            if input_text is None:
                continue  # Skip None test as it would cause TypeError before reaching our code
                
            try:
                result = self.calculator.calculate_bleu_score(
                    str(input_text), ["reference"], threshold=0.8
                )
                assert isinstance(result, EvaluationResult)
                assert result.metric_name == "BLEU"
            except Exception:
                pytest.fail(f"Evaluation should handle input gracefully: {type(input_text)}")


def test_metrics_calculator_spacy_unavailable():
    """Test MetricsCalculator when spaCy is unavailable."""
    with patch('src.evaluation.metrics.spacy.load') as mock_spacy:
        mock_spacy.side_effect = OSError("Model not found")
        
        calculator = MetricsCalculator()
        assert calculator.nlp is None
        
        # Should still work with fallback methods
        result = calculator.calculate_redundancy_score("Test sentence. Another sentence.")
        assert isinstance(result, EvaluationResult)