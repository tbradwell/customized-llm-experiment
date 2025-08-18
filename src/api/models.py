"""Pydantic models for API request/response schemas."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ContractType(str, Enum):
    """Supported contract types."""
    SERVICE_AGREEMENT = "service_agreement"
    EMPLOYMENT_CONTRACT = "employment_contract"
    NDA = "nda"
    CONSULTING_AGREEMENT = "consulting_agreement"
    GENERAL = "general"


class QualityGateStatus(str, Enum):
    """Quality gate status options."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    RETRY = "retry"


class ContractGenerationRequest(BaseModel):
    """Request model for contract generation."""
    contract_data: Dict[str, Any] = Field(
        ..., 
        description="Contract data including client info, terms, etc."
    )
    checklist: Optional[List[str]] = Field(
        default=None,
        description="Optional checklist of requirements to validate"
    )
    quality_threshold: Optional[float] = Field(
        default=4.5,
        ge=1.0,
        le=5.0,
        description="Minimum acceptable overall quality score"
    )
    max_iterations: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum regeneration attempts"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "contract_data": {
                    "contract_type": "service_agreement",
                    "client_name": "ABC Corporation",
                    "client_address": "123 Business St, City, State 12345",
                    "provider_name": "XYZ Legal Services",
                    "service_description": "Comprehensive legal consultation services",
                    "contract_value": "$50,000",
                    "start_date": "2024-02-01",
                    "end_date": "2024-12-31",
                    "payment_terms": "Net 30 days"
                },
                "checklist": [
                    "Include confidentiality clause",
                    "Specify termination conditions",
                    "Define scope of work clearly"
                ],
                "quality_threshold": 4.5,
                "max_iterations": 3
            }
        }


class QualityGateResult(BaseModel):
    """Quality gate evaluation result."""
    gate_name: str
    status: QualityGateStatus
    score: float
    threshold: float
    details: Dict[str, Any]
    recommendations: List[str]


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for contract quality."""
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None
    meteor_score: Optional[float] = None
    comet_score: Optional[float] = None
    llm_judge_score: Optional[float] = None
    redundancy_score: Optional[float] = None
    completeness_score: Optional[float] = None
    overall_quality_score: float


class ContractGenerationResponse(BaseModel):
    """Response model for contract generation."""
    success: bool
    contract_id: str
    quality_score: float
    meets_threshold: bool
    download_url: Optional[str] = None
    quality_report_url: Optional[str] = None
    evaluation_metrics: EvaluationMetrics
    quality_gates: List[QualityGateResult]
    iterations: int
    generation_time: float
    warnings: List[str]
    metadata: Dict[str, Any]


class ContractEvaluationRequest(BaseModel):
    """Request model for contract evaluation only."""
    contract_text: str = Field(..., description="Contract text to evaluate")
    reference_contracts: Optional[List[str]] = Field(
        default=None,
        description="Optional reference contracts for comparison"
    )
    contract_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context about the contract"
    )


class ContractEvaluationResponse(BaseModel):
    """Response model for contract evaluation."""
    success: bool
    evaluation_metrics: EvaluationMetrics
    quality_gates: List[QualityGateResult]
    overall_assessment: str
    recommendations: List[str]
    evaluation_time: float


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: bool = True
    message: str
    error_type: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime


class BatchGenerationRequest(BaseModel):
    """Request model for batch contract generation."""
    contracts: List[ContractGenerationRequest] = Field(
        ...,
        description="List of contract generation requests"
    )
    parallel_processing: bool = Field(
        default=False,
        description="Whether to process contracts in parallel"
    )


class BatchGenerationResponse(BaseModel):
    """Response model for batch contract generation."""
    success: bool
    total_contracts: int
    successful_contracts: int
    failed_contracts: int
    results: List[ContractGenerationResponse]
    batch_processing_time: float
    summary: Dict[str, Any]


class ExperimentTrackingResponse(BaseModel):
    """Response model for experiment tracking information."""
    experiment_id: str
    experiment_name: str
    run_id: str
    run_name: str
    metrics: Dict[str, float]
    parameters: Dict[str, str]
    artifacts: List[str]
    status: str


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics."""
    average_quality_score: float
    success_rate: float
    average_generation_time: float
    total_contracts_generated: int
    quality_gate_pass_rate: float
    metrics_breakdown: Dict[str, float]


class SystemStatusResponse(BaseModel):
    """System status response."""
    system_health: str
    active_experiments: int
    total_contracts_generated: int
    average_quality_score: float
    service_uptime: str
    last_updated: datetime


# Request/Response models for specific endpoints

class SkeletonUploadResponse(BaseModel):
    """Response for skeleton upload."""
    success: bool
    skeleton_id: str
    filename: str
    placeholders_found: List[str]
    validation_results: Dict[str, Any]


class ReferenceContractsUploadResponse(BaseModel):
    """Response for reference contracts upload."""
    success: bool
    reference_ids: List[str]
    contracts_processed: int
    total_text_length: int
    validation_results: List[Dict[str, Any]]


class ContractDownloadResponse(BaseModel):
    """Response for contract download."""
    contract_id: str
    filename: str
    file_size: int
    content_type: str
    generated_at: datetime


class QualityReportResponse(BaseModel):
    """Detailed quality report response."""
    contract_id: str
    overall_quality_score: float
    detailed_metrics: Dict[str, Any]
    quality_gates_summary: Dict[str, int]
    recommendations: List[str]
    areas_for_improvement: List[str]
    strengths: List[str]
    generation_metadata: Dict[str, Any]
    report_generated_at: datetime
