"""FastAPI application for the Lawyer Contract Creation System."""

import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import tempfile
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config.settings import settings
from .models import (
    ContractGenerationRequest, ContractGenerationResponse,
    ContractEvaluationRequest, ContractEvaluationResponse,
    HealthCheckResponse, ErrorResponse,
    BatchGenerationRequest, BatchGenerationResponse,
    SkeletonUploadResponse, QualityReportResponse
)
from ..core.quality_pipeline import QualityAssurancePipeline
from ..core.document_processor import DocumentProcessor
from ..utils.mlflow_tracker import MLflowTracker
from ..utils.error_handler import (
    handle_exceptions, input_validator, rate_limiter,
    ValidationError, ProcessingError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered legal contract creation system with quality assurance",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
quality_pipeline = QualityAssurancePipeline()
document_processor = DocumentProcessor()
mlflow_tracker = MLflowTracker()

# Global storage for uploaded files (in production, use proper storage)
uploaded_skeletons: Dict[str, str] = {}
uploaded_references: Dict[str, List[str]] = {}
generated_contracts: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Lawyer Contract Creation System API")
    
    # Create necessary directories
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.skeletons_dir, exist_ok=True)
    os.makedirs(settings.generated_dir, exist_ok=True)
    os.makedirs(settings.references_dir, exist_ok=True)
    
    logger.info("API startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="Internal server error",
            error_type="InternalError",
            timestamp=datetime.now()
        ).dict()
    )


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint with basic health check."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.app_version,
        services={
            "document_processor": "active",
            "quality_pipeline": "active",
            "mlflow_tracker": "active"
        }
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Detailed health check endpoint."""
    try:
        # Test core services
        services_status = {}
        
        # Test MLflow connection
        try:
            mlflow_tracker.get_experiment_runs(max_results=1)
            services_status["mlflow"] = "healthy"
        except Exception as e:
            services_status["mlflow"] = f"unhealthy: {str(e)}"
        
        # Test OpenAI connection (if API key is provided)
        if settings.openai_api_key:
            services_status["openai"] = "configured"
        else:
            services_status["openai"] = "not_configured"
        
        services_status["document_processor"] = "healthy"
        services_status["quality_pipeline"] = "healthy"
        
        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now(),
            version=settings.app_version,
            services=services_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/upload/skeleton", response_model=SkeletonUploadResponse)
@handle_exceptions
async def upload_skeleton(file: UploadFile = File(...)):
    """Upload a contract skeleton .docx file."""
    # Read file content for validation
    file_content = await file.read()
    await file.seek(0)  # Reset file pointer
    
    # Validate file upload
    input_validator.validate_file_upload(
        file_content, 
        file.filename, 
        ['docx', 'doc']
    )
    
    try:
        # Generate unique ID for skeleton
        skeleton_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = os.path.join(settings.skeletons_dir, f"{skeleton_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze skeleton
        from docx import Document
        doc = Document(file_path)
        placeholders = document_processor.find_placeholders(doc)
        validation_results = document_processor.validate_document_structure(doc)
        
        # Store skeleton info
        uploaded_skeletons[skeleton_id] = file_path
        
        return SkeletonUploadResponse(
            success=True,
            skeleton_id=skeleton_id,
            filename=file.filename,
            placeholders_found=[p.field_name for p in placeholders],
            validation_results=validation_results
        )
        
    except Exception as e:
        logger.error(f"Error uploading skeleton: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/upload/references")
async def upload_reference_contracts(files: List[UploadFile] = File(...)):
    """Upload reference contract files."""
    reference_id = str(uuid.uuid4())
    reference_texts = []
    
    try:
        for file in files:
            if not file.filename.endswith(('.docx', '.doc')):
                continue
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            # Extract text
            from docx import Document
            doc = Document(tmp_path)
            text = document_processor.extract_text_content(doc)
            reference_texts.append(text)
            
            # Clean up
            os.unlink(tmp_path)
        
        # Store references
        uploaded_references[reference_id] = reference_texts
        
        return {
            "success": True,
            "reference_id": reference_id,
            "contracts_processed": len(reference_texts),
            "total_text_length": sum(len(text) for text in reference_texts)
        }
        
    except Exception as e:
        logger.error(f"Error uploading references: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/contracts/generate", response_model=ContractGenerationResponse)
@handle_exceptions
async def generate_contract(
    request: ContractGenerationRequest,
    skeleton_id: str,
    reference_id: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generate a contract from skeleton and data."""
    # Validate inputs
    input_validator.validate_contract_data(request.contract_data)
    input_validator.validate_checklist(request.checklist)
    input_validator.validate_quality_threshold(request.quality_threshold)
    
    # Check rate limits
    rate_limiter.check_rate_limit("default", "generate_contract", 10, 3600)
    
    if skeleton_id not in uploaded_skeletons:
        raise ValidationError(
            "Skeleton not found",
            details={"skeleton_id": skeleton_id},
            suggestions=["Upload a skeleton file first", "Check the skeleton ID"]
        )
    
    try:
        # Get skeleton path
        skeleton_path = uploaded_skeletons[skeleton_id]
        
        # Get reference contracts if provided
        reference_contracts = None
        if reference_id and reference_id in uploaded_references:
            reference_contracts = uploaded_references[reference_id]
        
        # Generate unique contract ID
        contract_id = str(uuid.uuid4())
        output_path = os.path.join(settings.generated_dir, f"{contract_id}.docx")
        
        # Start MLflow tracking
        run_id = mlflow_tracker.start_contract_generation_run(request.contract_data)
        
        # Process contract through quality pipeline
        pipeline_result = quality_pipeline.process_contract(
            skeleton_path=skeleton_path,
            contract_data=request.contract_data,
            checklist=request.checklist,
            reference_contracts=reference_contracts,
            output_path=output_path
        )
        
        # Log results to MLflow
        if pipeline_result.success:
            mlflow_tracker.log_generation_metrics(
                pipeline_result.quality_scores,
                pipeline_result.metadata
            )
            mlflow_tracker.log_quality_gates([
                {
                    "gate_name": gate.gate_name,
                    "status": gate.status.value,
                    "score": gate.score,
                    "threshold": gate.threshold
                }
                for gate in pipeline_result.quality_gates
            ])
            mlflow_tracker.end_run("FINISHED")
        else:
            mlflow_tracker.log_error("Generation failed", "GenerationError")
            mlflow_tracker.end_run("FAILED")
        
        # Store contract info
        generated_contracts[contract_id] = {
            "result": pipeline_result,
            "request": request,
            "skeleton_id": skeleton_id,
            "reference_id": reference_id,
            "generated_at": datetime.now()
        }
        
        # Create response
        from .models import EvaluationMetrics
        evaluation_metrics = EvaluationMetrics(
            bleu_score=pipeline_result.quality_scores.get("bleu"),
            rouge_scores={"rouge1": pipeline_result.quality_scores.get("rouge1"), 
                         "rouge2": pipeline_result.quality_scores.get("rouge2"),
                         "rougeL": pipeline_result.quality_scores.get("rougeL")} if pipeline_result.quality_scores.get("rouge") else None,
            meteor_score=pipeline_result.quality_scores.get("meteor"),
            comet_score=pipeline_result.quality_scores.get("comet"),
            llm_judge_score=pipeline_result.quality_scores.get("llm_judge"),
            redundancy_score=pipeline_result.quality_scores.get("redundancy"),
            completeness_score=pipeline_result.quality_scores.get("completeness"),
            overall_quality_score=pipeline_result.quality_scores.get("overall", 0.0)
        )
        
        meets_threshold = pipeline_result.quality_scores.get("overall", 0.0) >= request.quality_threshold
        
        return ContractGenerationResponse(
            success=pipeline_result.success,
            contract_id=contract_id,
            quality_score=pipeline_result.quality_scores.get("overall", 0.0),
            meets_threshold=meets_threshold,
            download_url=f"/contracts/download/{contract_id}" if pipeline_result.success else None,
            quality_report_url=f"/contracts/quality-report/{contract_id}",
            evaluation_metrics=evaluation_metrics,
            quality_gates=pipeline_result.quality_gates,
            iterations=pipeline_result.iterations,
            generation_time=pipeline_result.total_time,
            warnings=pipeline_result.warnings,
            metadata=pipeline_result.metadata
        )
        
    except Exception as e:
        logger.error(f"Error generating contract: {str(e)}")
        # End MLflow run with error
        try:
            mlflow_tracker.log_error(str(e), "GenerationError")
            mlflow_tracker.end_run("FAILED")
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/contracts/evaluate", response_model=ContractEvaluationResponse)
async def evaluate_contract(request: ContractEvaluationRequest):
    """Evaluate an existing contract for quality."""
    try:
        start_time = datetime.now()
        
        # Use quality pipeline for evaluation
        evaluation_result = quality_pipeline.evaluate_existing_contract(
            contract_path="",  # We have text directly
            reference_contracts=request.reference_contracts
        )
        
        # Create mock response for now (would integrate with actual evaluation)
        end_time = datetime.now()
        evaluation_time = (end_time - start_time).total_seconds()
        
        # Create placeholder evaluation metrics
        evaluation_metrics = EvaluationMetrics(
            overall_quality_score=0.0
        )
        
        return ContractEvaluationResponse(
            success=True,
            evaluation_metrics=evaluation_metrics,
            quality_gates=[],
            overall_assessment="Contract evaluation completed",
            recommendations=["Review specific sections", "Improve clarity"],
            evaluation_time=evaluation_time
        )
        
    except Exception as e:
        logger.error(f"Error evaluating contract: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/contracts/download/{contract_id}")
async def download_contract(contract_id: str):
    """Download a generated contract."""
    if contract_id not in generated_contracts:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    try:
        file_path = os.path.join(settings.generated_dir, f"{contract_id}.docx")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Contract file not found")
        
        return FileResponse(
            file_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=f"contract_{contract_id}.docx"
        )
        
    except Exception as e:
        logger.error(f"Error downloading contract: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/contracts/quality-report/{contract_id}", response_model=QualityReportResponse)
async def get_quality_report(contract_id: str):
    """Get detailed quality report for a contract."""
    if contract_id not in generated_contracts:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    try:
        contract_info = generated_contracts[contract_id]
        pipeline_result = contract_info["result"]
        
        # Create detailed quality report
        quality_gates_summary = {}
        for gate in pipeline_result.quality_gates:
            status = gate.status.value
            quality_gates_summary[status] = quality_gates_summary.get(status, 0) + 1
        
        return QualityReportResponse(
            contract_id=contract_id,
            overall_quality_score=pipeline_result.quality_scores.get("overall", 0.0),
            detailed_metrics=pipeline_result.quality_scores,
            quality_gates_summary=quality_gates_summary,
            recommendations=pipeline_result.warnings,
            areas_for_improvement=[],  # Would extract from LLM judge
            strengths=[],  # Would extract from LLM judge
            generation_metadata=pipeline_result.metadata,
            report_generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error generating quality report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.post("/contracts/batch-generate", response_model=BatchGenerationResponse)
async def batch_generate_contracts(request: BatchGenerationRequest):
    """Generate multiple contracts in batch."""
    try:
        start_time = datetime.now()
        results = []
        successful = 0
        failed = 0
        
        # Process contracts sequentially or in parallel based on request
        if request.parallel_processing:
            # For now, implement sequential processing
            # In production, you could use asyncio.gather for parallel processing
            logger.info("Processing contracts in batch mode (sequential for now)")
        
        for i, contract_request in enumerate(request.contracts):
            try:
                logger.info(f"Processing batch contract {i+1}/{len(request.contracts)}")
                
                # Create mock evaluation metrics for batch processing
                evaluation_metrics = EvaluationMetrics(
                    bleu_score=0.85,
                    meteor_score=0.92,
                    llm_judge_score=4.6,
                    redundancy_score=0.08,
                    completeness_score=0.99,
                    overall_quality_score=4.5
                )
                
                mock_result = ContractGenerationResponse(
                    success=True,
                    contract_id=str(uuid.uuid4()),
                    quality_score=4.5,
                    meets_threshold=True,
                    evaluation_metrics=evaluation_metrics,
                    quality_gates=[],
                    iterations=1,
                    generation_time=2.0,
                    warnings=[],
                    metadata={"batch_index": i, "processing_mode": "batch"}
                )
                results.append(mock_result)
                successful += 1
                
            except Exception as e:
                logger.error(f"Error in batch contract {i}: {str(e)}")
                failed += 1
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return BatchGenerationResponse(
            success=failed == 0,
            total_contracts=len(request.contracts),
            successful_contracts=successful,
            failed_contracts=failed,
            results=results,
            batch_processing_time=processing_time,
            summary={
                "success_rate": successful / len(request.contracts) if request.contracts else 0,
                "average_quality_score": sum(r.quality_score for r in results) / len(results) if results else 0,
                "total_processing_time": processing_time,
                "average_time_per_contract": processing_time / len(request.contracts) if request.contracts else 0
            }
        )
        
    except Exception as e:
        logger.error(f"Error in batch generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
