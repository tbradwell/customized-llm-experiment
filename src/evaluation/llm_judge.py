"""LLM Judge system for holistic contract quality evaluation."""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

from openai import OpenAI
from config.settings import settings
from .metrics import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class JudgmentCriteria:
    """Criteria for LLM judgment."""
    name: str
    description: str
    weight: float
    max_score: int = 5


class LLMJudge:
    """LLM-based quality judge for comprehensive contract evaluation."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        
        # Define evaluation criteria
        self.criteria = [
            JudgmentCriteria(
                name="legal_accuracy",
                description="Legal accuracy and correctness of contract terms and clauses",
                weight=0.3
            ),
            JudgmentCriteria(
                name="completeness",
                description="Completeness of contract sections and required information",
                weight=0.25
            ),
            JudgmentCriteria(
                name="professional_language",
                description="Use of appropriate legal terminology and professional language",
                weight=0.2
            ),
            JudgmentCriteria(
                name="logical_consistency",
                description="Internal consistency and logical flow of contract terms",
                weight=0.15
            ),
            JudgmentCriteria(
                name="clarity_readability",
                description="Clarity, readability, and overall document structure",
                weight=0.1
            )
        ]
    
    def evaluate_contract(self, generated_contract: str, 
                         contract_context: Dict[str, Any],
                         reference_contracts: Optional[List[str]] = None,
                         threshold: float = 4.5) -> EvaluationResult:
        """Evaluate contract using LLM judge.
        
        Args:
            generated_contract: The generated contract text
            contract_context: Context information about the contract
            reference_contracts: Optional reference contracts for comparison
            threshold: Minimum acceptable LLM judge score
            
        Returns:
            EvaluationResult containing LLM judge assessment
        """
        try:
            # Prepare evaluation prompt
            evaluation_prompt = self._create_evaluation_prompt(
                generated_contract, contract_context, reference_contracts
            )
            
            # Get LLM evaluation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=2000
            )
            
            # Parse response
            evaluation_result = self._parse_llm_response(response.choices[0].message.content)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(evaluation_result["criteria_scores"])
            
            details = {
                "criteria_scores": evaluation_result["criteria_scores"],
                "detailed_feedback": evaluation_result["feedback"],
                "recommendations": evaluation_result["recommendations"],
                "model_used": self.model,
                "evaluation_method": "structured_criteria"
            }
            
            return EvaluationResult(
                metric_name="LLM_Judge",
                score=overall_score,
                details=details,
                passed_threshold=overall_score >= threshold,
                threshold=threshold
            )
            
        except Exception as e:
            logger.error(f"Error in LLM judge evaluation: {str(e)}")
            return EvaluationResult(
                metric_name="LLM_Judge",
                score=0.0,
                details={"error": str(e)},
                passed_threshold=False,
                threshold=threshold
            )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM judge."""
        return """You are an expert legal contract evaluator with extensive experience in contract law and document quality assessment. Your task is to evaluate generated legal contracts for quality, accuracy, and completeness.

You will evaluate contracts based on the following criteria:
1. Legal Accuracy (30%): Correctness of legal terms, clauses, and overall legal soundness
2. Completeness (25%): Presence of all necessary contract elements and information
3. Professional Language (20%): Appropriate legal terminology and professional writing style
4. Logical Consistency (15%): Internal consistency and logical flow of contract terms
5. Clarity & Readability (10%): Clear structure and readability

For each criterion, provide:
- A score from 1-5 (1=Poor, 2=Below Average, 3=Average, 4=Good, 5=Excellent)
- Specific feedback explaining the score
- Recommendations for improvement if applicable

Respond in JSON format with the following structure:
{
    "criteria_scores": {
        "legal_accuracy": {"score": X, "feedback": "..."},
        "completeness": {"score": X, "feedback": "..."},
        "professional_language": {"score": X, "feedback": "..."},
        "logical_consistency": {"score": X, "feedback": "..."},
        "clarity_readability": {"score": X, "feedback": "..."}
    },
    "overall_assessment": "...",
    "key_strengths": ["...", "..."],
    "areas_for_improvement": ["...", "..."],
    "recommendations": ["...", "..."]
}

Be thorough, objective, and provide constructive feedback that would help improve contract quality."""
    
    def _create_evaluation_prompt(self, generated_contract: str, 
                                 contract_context: Dict[str, Any],
                                 reference_contracts: Optional[List[str]] = None) -> str:
        """Create the evaluation prompt for the LLM judge."""
        
        prompt_parts = [
            "Please evaluate the following generated legal contract:",
            "",
            "=== GENERATED CONTRACT ===",
            generated_contract,
            "",
            "=== CONTRACT CONTEXT ===",
            f"Contract Type: {contract_context.get('contract_type', 'Unknown')}",
            f"Client: {contract_context.get('client_name', 'N/A')}",
            f"Provider: {contract_context.get('provider_name', 'N/A')}",
        ]
        
        # Add additional context if available
        if 'service_description' in contract_context:
            prompt_parts.append(f"Service Description: {contract_context['service_description']}")
        
        if 'contract_value' in contract_context:
            prompt_parts.append(f"Contract Value: {contract_context['contract_value']}")
        
        # Add reference contracts if provided
        if reference_contracts:
            prompt_parts.extend([
                "",
                "=== REFERENCE CONTRACTS (for comparison) ===",
            ])
            for i, ref in enumerate(reference_contracts[:2], 1):  # Limit to 2 references
                prompt_parts.extend([
                    f"Reference {i}:",
                    ref[:1000] + "..." if len(ref) > 1000 else ref,  # Truncate if too long
                    ""
                ])
        
        prompt_parts.extend([
            "",
            "Please provide a comprehensive evaluation based on the criteria outlined in your instructions.",
            "Focus particularly on legal accuracy and completeness, as these are critical for contract quality."
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data."""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                evaluation_data = json.loads(json_text)
                
                # Validate structure
                if "criteria_scores" in evaluation_data:
                    return {
                        "criteria_scores": evaluation_data["criteria_scores"],
                        "feedback": evaluation_data.get("overall_assessment", ""),
                        "recommendations": evaluation_data.get("recommendations", []),
                        "strengths": evaluation_data.get("key_strengths", []),
                        "areas_for_improvement": evaluation_data.get("areas_for_improvement", [])
                    }
            
            # Fallback: try to parse without JSON structure
            return self._parse_unstructured_response(response_text)
            
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON, falling back to text parsing")
            return self._parse_unstructured_response(response_text)
    
    def _parse_unstructured_response(self, response_text: str) -> Dict[str, Any]:
        """Parse unstructured LLM response as fallback."""
        # Extract scores using regex or simple parsing
        criteria_scores = {}
        
        # Default scores if parsing fails
        for criteria in self.criteria:
            criteria_scores[criteria.name] = {
                "score": 3.0,  # Default to average
                "feedback": "Could not parse specific feedback"
            }
        
        return {
            "criteria_scores": criteria_scores,
            "feedback": response_text,
            "recommendations": ["Review LLM response format"],
            "strengths": [],
            "areas_for_improvement": ["Response parsing"]
        }
    
    def _calculate_overall_score(self, criteria_scores: Dict[str, Dict[str, Any]]) -> float:
        """Calculate weighted overall score from criteria scores."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for criteria in self.criteria:
            if criteria.name in criteria_scores:
                score = float(criteria_scores[criteria.name].get("score", 0))
                total_weighted_score += score * criteria.weight
                total_weight += criteria.weight
        
        if total_weight > 0:
            return total_weighted_score / total_weight
        else:
            return 0.0
    
    def evaluate_specific_aspect(self, contract_text: str, aspect: str, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a specific aspect of the contract.
        
        Args:
            contract_text: The contract text to evaluate
            aspect: Specific aspect to focus on (e.g., "legal_accuracy", "completeness")
            context: Contract context information
            
        Returns:
            Dictionary containing specific aspect evaluation
        """
        try:
            aspect_prompt = self._create_aspect_specific_prompt(contract_text, aspect, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are evaluating the {aspect} of a legal contract. Provide detailed analysis and a score from 1-5."
                    },
                    {
                        "role": "user",
                        "content": aspect_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return {
                "aspect": aspect,
                "evaluation": response.choices[0].message.content,
                "model_used": self.model
            }
            
        except Exception as e:
            logger.error(f"Error in aspect-specific evaluation: {str(e)}")
            return {
                "aspect": aspect,
                "evaluation": f"Error: {str(e)}",
                "error": True
            }
    
    def _create_aspect_specific_prompt(self, contract_text: str, aspect: str, 
                                     context: Dict[str, Any]) -> str:
        """Create a prompt for evaluating a specific aspect."""
        aspect_descriptions = {
            "legal_accuracy": "Focus on the legal correctness, proper use of legal terms, and enforceability of contract clauses.",
            "completeness": "Evaluate whether all necessary contract elements are present and properly detailed.",
            "professional_language": "Assess the use of appropriate legal terminology and professional writing style.",
            "logical_consistency": "Check for internal consistency and logical flow between different contract sections.",
            "clarity_readability": "Evaluate the clarity, structure, and overall readability of the contract."
        }
        
        description = aspect_descriptions.get(aspect, f"Evaluate the {aspect} of this contract.")
        
        return f"""
Please evaluate the {aspect} of the following contract:

{description}

Contract Context:
- Type: {context.get('contract_type', 'Unknown')}
- Client: {context.get('client_name', 'N/A')}
- Provider: {context.get('provider_name', 'N/A')}

Contract Text:
{contract_text}

Provide:
1. A score from 1-5 for {aspect}
2. Detailed explanation of your assessment
3. Specific examples from the contract
4. Recommendations for improvement if applicable
"""
    
    def batch_evaluate(self, contracts: List[str], contexts: List[Dict[str, Any]],
                      threshold: float = 4.5) -> List[EvaluationResult]:
        """Evaluate multiple contracts in batch.
        
        Args:
            contracts: List of contract texts to evaluate
            contexts: List of context dictionaries for each contract
            threshold: Minimum acceptable score
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        for i, (contract, context) in enumerate(zip(contracts, contexts)):
            logger.info(f"Evaluating contract {i+1}/{len(contracts)}")
            result = self.evaluate_contract(contract, context, threshold=threshold)
            results.append(result)
        
        return results
