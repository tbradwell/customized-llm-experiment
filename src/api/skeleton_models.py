"""API models for skeleton processor endpoints."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SkeletonProcessingRequest(BaseModel):
    """Request model for skeleton processing."""
    
    upload_id: str = Field(..., description="ID of uploaded documents")
    async_processing: bool = Field(default=True, description="Process in background")
    
    class Config:
        schema_extra = {
            "example": {
                "upload_id": "123e4567-e89b-12d3-a456-426614174000",
                "async_processing": True
            }
        }


class SkeletonProcessingResponse(BaseModel):
    """Response model for skeleton processing."""
    
    success: bool = Field(..., description="Whether processing was successful")
    job_id: str = Field(..., description="Job ID for tracking")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    
    # Optional fields for completed processing
    skeleton_id: Optional[str] = Field(None, description="Generated skeleton ID")
    template_download_url: Optional[str] = Field(None, description="URL to download template")
    stats_url: Optional[str] = Field(None, description="URL for detailed statistics")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    # Optional fields for async processing
    async_processing: Optional[bool] = Field(None, description="Whether processing is async")
    status_check_url: Optional[str] = Field(None, description="URL to check status")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "job_id": "456e7890-e89b-12d3-a456-426614174000",
                "status": "completed",
                "message": "Skeleton processing completed successfully",
                "skeleton_id": "789e0123-e89b-12d3-a456-426614174000",
                "template_download_url": "/skeleton/download/789e0123-e89b-12d3-a456-426614174000",
                "stats_url": "/skeleton/stats/456e7890-e89b-12d3-a456-426614174000",
                "processing_time": 45.2
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    
    success: bool = Field(..., description="Whether upload was successful")
    upload_id: str = Field(..., description="Unique upload identifier")
    documents_count: int = Field(..., description="Number of documents uploaded")
    filenames: List[str] = Field(..., description="List of uploaded filenames")
    validation_passed: bool = Field(..., description="Whether validation passed")
    validation_issues: List[str] = Field(default=[], description="Validation issues found")
    total_size_mb: float = Field(..., description="Total size of uploaded files in MB")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "upload_id": "123e4567-e89b-12d3-a456-426614174000",
                "documents_count": 3,
                "filenames": ["contract1.docx", "contract2.docx", "contract3.docx"],
                "validation_passed": True,
                "validation_issues": [],
                "total_size_mb": 2.4
            }
        }


class SkeletonValidationResponse(BaseModel):
    """Response model for skeleton validation."""
    
    success: bool = Field(..., description="Whether validation was successful")
    valid_skeleton: bool = Field(..., description="Whether skeleton is valid")
    validation_results: Dict[str, Any] = Field(..., description="Detailed validation results")
    issues: List[str] = Field(default=[], description="Issues found during validation")
    filename: str = Field(..., description="Name of validated file")
    file_size_mb: float = Field(..., description="File size in MB")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "valid_skeleton": True,
                "validation_results": {
                    "file_format": "valid",
                    "paragraph_count": 15,
                    "delimiter_counts": {
                        "block_start": 3,
                        "block_end": 3,
                        "uncertain": 2
                    },
                    "balanced_delimiters": True,
                    "has_content": True
                },
                "issues": [],
                "filename": "skeleton_template.docx",
                "file_size_mb": 0.8
            }
        }


class SkeletonStatsResponse(BaseModel):
    """Response model for skeleton processing statistics."""
    
    job_id: str = Field(..., description="Processing job ID")
    skeleton_id: str = Field(..., description="Generated skeleton ID")
    processing_completed_at: datetime = Field(..., description="When processing completed")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    
    input_statistics: Dict[str, Any] = Field(..., description="Input document statistics")
    algorithm_parameters: Dict[str, Any] = Field(..., description="Algorithm parameters used")
    clustering_statistics: Dict[str, Any] = Field(..., description="Clustering results")
    output_information: Dict[str, Any] = Field(..., description="Output file information")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "456e7890-e89b-12d3-a456-426614174000",
                "skeleton_id": "789e0123-e89b-12d3-a456-426614174000",
                "processing_completed_at": "2024-01-15T10:30:45",
                "total_processing_time": 45.2,
                "input_statistics": {
                    "document_count": 3,
                    "total_paragraphs": 87,
                    "total_clusters": 12
                },
                "algorithm_parameters": {
                    "embedding_model": "text-embedding-3-large",
                    "high_homogeneity_threshold": 0.8,
                    "clustering_algorithm": "Spherical K-means"
                },
                "clustering_statistics": {
                    "clusters_created": 12,
                    "blocks_generated": 8,
                    "uncertain_blocks": 4
                },
                "output_information": {
                    "skeleton_id": "789e0123-e89b-12d3-a456-426614174000",
                    "template_path": "skeleton_template_789e0123.docx",
                    "delimiter_positions": 15,
                    "created_at": "2024-01-15T10:30:45"
                }
            }
        }


class SkeletonProcessingStatus(BaseModel):
    """Model for processing status updates."""
    
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current status")
    progress: int = Field(..., description="Progress percentage (0-100)")
    started_at: datetime = Field(..., description="When processing started")
    document_count: int = Field(..., description="Number of documents being processed")
    
    # Optional fields based on status
    completed_at: Optional[datetime] = Field(None, description="When processing completed")
    failed_at: Optional[datetime] = Field(None, description="When processing failed")
    error: Optional[str] = Field(None, description="Error message if failed")
    skeleton_id: Optional[str] = Field(None, description="Generated skeleton ID if completed")
    template_download_url: Optional[str] = Field(None, description="Download URL if completed")
    stats_url: Optional[str] = Field(None, description="Stats URL if completed")
    processing_time: Optional[float] = Field(None, description="Total processing time if completed")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "456e7890-e89b-12d3-a456-426614174000",
                "status": "processing",
                "progress": 65,
                "started_at": "2024-01-15T10:25:30",
                "document_count": 3
            }
        }


class SkeletonJobSummary(BaseModel):
    """Summary model for skeleton processing jobs."""
    
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    started_at: datetime = Field(..., description="When job started")
    document_count: int = Field(..., description="Number of documents")
    
    # Optional fields based on status
    completed_at: Optional[datetime] = Field(None, description="When completed")
    failed_at: Optional[datetime] = Field(None, description="When failed")
    skeleton_id: Optional[str] = Field(None, description="Generated skeleton ID")
    error: Optional[str] = Field(None, description="Error message")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "456e7890-e89b-12d3-a456-426614174000",
                "status": "completed",
                "started_at": "2024-01-15T10:25:30",
                "document_count": 3,
                "completed_at": "2024-01-15T10:30:45",
                "skeleton_id": "789e0123-e89b-12d3-a456-426614174000"
            }
        }


class SkeletonJobsListResponse(BaseModel):
    """Response model for listing skeleton processing jobs."""
    
    total_jobs: int = Field(..., description="Total number of jobs")
    jobs: List[SkeletonJobSummary] = Field(..., description="List of job summaries")
    
    class Config:
        schema_extra = {
            "example": {
                "total_jobs": 2,
                "jobs": [
                    {
                        "job_id": "456e7890-e89b-12d3-a456-426614174000",
                        "status": "completed",
                        "started_at": "2024-01-15T10:25:30",
                        "document_count": 3,
                        "completed_at": "2024-01-15T10:30:45",
                        "skeleton_id": "789e0123-e89b-12d3-a456-426614174000"
                    },
                    {
                        "job_id": "abc12345-e89b-12d3-a456-426614174000",
                        "status": "processing",
                        "started_at": "2024-01-15T11:00:00",
                        "document_count": 5
                    }
                ]
            }
        }


class SkeletonHealthResponse(BaseModel):
    """Response model for skeleton processor health check."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    services: Dict[str, str] = Field(..., description="Status of individual services")
    active_jobs: int = Field(..., description="Number of active processing jobs")
    uploaded_document_sets: int = Field(..., description="Number of uploaded document sets")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T12:00:00",
                "services": {
                    "skeleton_processor": "healthy",
                    "openai_embeddings": "healthy",
                    "text_cleaner": "healthy",
                    "clustering_engine": "healthy",
                    "skeleton_generator": "healthy"
                },
                "active_jobs": 2,
                "uploaded_document_sets": 5
            }
        }


class SkeletonConfigurationResponse(BaseModel):
    """Response model for skeleton processor configuration."""
    
    embedding_model: str = Field(..., description="OpenAI embedding model in use")
    high_homogeneity_threshold: float = Field(..., description="High homogeneity threshold")
    low_homogeneity_threshold: float = Field(..., description="Low homogeneity threshold")
    language: str = Field(..., description="Processing language")
    clustering_algorithm: str = Field(..., description="Clustering algorithm")
    distance_metric: str = Field(..., description="Distance metric used")
    
    class Config:
        schema_extra = {
            "example": {
                "embedding_model": "text-embedding-3-large",
                "high_homogeneity_threshold": 0.8,
                "low_homogeneity_threshold": 0.8,
                "language": "he",
                "clustering_algorithm": "Spherical K-means",
                "distance_metric": "Cosine similarity"
            }
        }
