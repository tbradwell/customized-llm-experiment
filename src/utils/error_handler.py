"""Centralized error handling and validation utilities."""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur in the system."""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    OPENAI_API_ERROR = "openai_api_error"
    DOCUMENT_ERROR = "document_error"
    QUALITY_GATE_ERROR = "quality_gate_error"
    MLFLOW_ERROR = "mlflow_error"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class ErrorDetail:
    """Detailed error information."""
    error_type: ErrorType
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None


class ValidationError(Exception):
    """Custom validation error."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None, suggestions: List[str] = None):
        self.message = message
        self.details = details or {}
        self.suggestions = suggestions or []
        super().__init__(self.message)


class ProcessingError(Exception):
    """Custom processing error."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None, suggestions: List[str] = None):
        self.message = message
        self.details = details or {}
        self.suggestions = suggestions or []
        super().__init__(self.message)


class QualityGateError(Exception):
    """Custom quality gate error."""
    
    def __init__(self, message: str, failed_gates: List[str] = None, details: Dict[str, Any] = None):
        self.message = message
        self.failed_gates = failed_gates or []
        self.details = details or {}
        super().__init__(self.message)


class ErrorHandler:
    """Centralized error handling and logging."""
    
    def __init__(self):
        self.error_log: List[ErrorDetail] = []
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorDetail:
        """Handle and log an error."""
        error_detail = self._create_error_detail(error, context)
        self._log_error(error_detail)
        self.error_log.append(error_detail)
        return error_detail
    
    def _create_error_detail(self, error: Exception, context: Dict[str, Any] = None) -> ErrorDetail:
        """Create detailed error information."""
        error_type = self._determine_error_type(error)
        
        details = {
            "exception_type": type(error).__name__,
            "stack_trace": str(error)
        }
        
        suggestions = []
        
        # Add specific details and suggestions based on error type
        if isinstance(error, ValidationError):
            details.update(error.details)
            suggestions.extend(error.suggestions)
        elif isinstance(error, ProcessingError):
            details.update(error.details)
            suggestions.extend(error.suggestions)
        elif isinstance(error, QualityGateError):
            details["failed_gates"] = error.failed_gates
            details.update(error.details)
            suggestions.append("Review quality thresholds and contract content")
        elif "openai" in str(error).lower():
            suggestions.extend([
                "Check OpenAI API key configuration",
                "Verify API rate limits",
                "Check network connectivity"
            ])
        elif "mlflow" in str(error).lower():
            suggestions.extend([
                "Check MLflow tracking URI",
                "Verify MLflow server is running",
                "Check database connectivity"
            ])
        
        return ErrorDetail(
            error_type=error_type,
            message=str(error),
            details=details,
            timestamp=datetime.now(),
            context=context,
            suggestions=suggestions
        )
    
    def _determine_error_type(self, error: Exception) -> ErrorType:
        """Determine the type of error."""
        if isinstance(error, ValidationError):
            return ErrorType.VALIDATION_ERROR
        elif isinstance(error, ProcessingError):
            return ErrorType.PROCESSING_ERROR
        elif isinstance(error, QualityGateError):
            return ErrorType.QUALITY_GATE_ERROR
        elif "openai" in str(error).lower():
            return ErrorType.OPENAI_API_ERROR
        elif "mlflow" in str(error).lower():
            return ErrorType.MLFLOW_ERROR
        elif "docx" in str(error).lower() or "document" in str(error).lower():
            return ErrorType.DOCUMENT_ERROR
        elif "config" in str(error).lower() or "setting" in str(error).lower():
            return ErrorType.CONFIGURATION_ERROR
        else:
            return ErrorType.SYSTEM_ERROR
    
    def _log_error(self, error_detail: ErrorDetail):
        """Log error details."""
        logger.error(
            f"Error [{error_detail.error_type.value}]: {error_detail.message}",
            extra={
                "error_type": error_detail.error_type.value,
                "details": error_detail.details,
                "context": error_detail.context,
                "suggestions": error_detail.suggestions,
                "timestamp": error_detail.timestamp.isoformat()
            }
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        error_counts = {}
        for error in self.error_log:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "error_counts": error_counts,
            "last_error": self.error_log[-1] if self.error_log else None,
            "generated_at": datetime.now().isoformat()
        }


class InputValidator:
    """Validates input data for various components."""
    
    @staticmethod
    def validate_contract_data(contract_data: Dict[str, Any]) -> None:
        """Validate contract data input."""
        if not contract_data:
            raise ValidationError(
                "Contract data cannot be empty",
                details={"provided_data": contract_data},
                suggestions=["Provide contract data with required fields"]
            )
        
        required_fields = ["client_name", "provider_name"]
        missing_fields = [field for field in required_fields if not contract_data.get(field)]
        
        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing_fields)}",
                details={
                    "missing_fields": missing_fields,
                    "provided_fields": list(contract_data.keys())
                },
                suggestions=[f"Add {field} to contract data" for field in missing_fields]
            )
        
        # Validate field types and content
        string_fields = ["client_name", "provider_name", "service_description"]
        for field in string_fields:
            if field in contract_data:
                value = contract_data[field]
                if not isinstance(value, str) or len(value.strip()) == 0:
                    raise ValidationError(
                        f"Field '{field}' must be a non-empty string",
                        details={"field": field, "value": value, "type": type(value).__name__},
                        suggestions=[f"Provide a valid string value for {field}"]
                    )
                
                # Check for potentially malicious content
                dangerous_patterns = ["<script", "javascript:", "data:"]
                if any(pattern in value.lower() for pattern in dangerous_patterns):
                    raise ValidationError(
                        f"Field '{field}' contains potentially dangerous content",
                        details={"field": field, "value": value},
                        suggestions=["Remove script tags and suspicious content"]
                    )
        
        # Validate monetary values
        if "contract_value" in contract_data:
            value = contract_data["contract_value"]
            if isinstance(value, str):
                # Remove currency symbols and whitespace
                numeric_value = value.replace("$", "").replace(",", "").strip()
                try:
                    float(numeric_value)
                except ValueError:
                    raise ValidationError(
                        f"Invalid contract value format: {value}",
                        details={"field": "contract_value", "value": value},
                        suggestions=["Use format like '$50,000' or '50000'"]
                    )
        
        # Validate dates
        date_fields = ["start_date", "end_date"]
        for field in date_fields:
            if field in contract_data:
                date_value = contract_data[field]
                if isinstance(date_value, str):
                    # Basic date format validation
                    import re
                    date_patterns = [
                        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                        r'\d{2}-\d{2}-\d{4}'   # MM-DD-YYYY
                    ]
                    
                    if not any(re.match(pattern, date_value) for pattern in date_patterns):
                        raise ValidationError(
                            f"Invalid date format for {field}: {date_value}",
                            details={"field": field, "value": date_value},
                            suggestions=["Use format YYYY-MM-DD, MM/DD/YYYY, or MM-DD-YYYY"]
                        )
    
    @staticmethod
    def validate_checklist(checklist: List[str]) -> None:
        """Validate checklist input."""
        if checklist is not None:
            if not isinstance(checklist, list):
                raise ValidationError(
                    "Checklist must be a list of strings",
                    details={"provided_type": type(checklist).__name__},
                    suggestions=["Provide checklist as a list of requirement strings"]
                )
            
            for i, item in enumerate(checklist):
                if not isinstance(item, str) or len(item.strip()) == 0:
                    raise ValidationError(
                        f"Checklist item {i} must be a non-empty string",
                        details={"item_index": i, "item_value": item},
                        suggestions=["Ensure all checklist items are meaningful text"]
                    )
                
                if len(item) > 500:
                    raise ValidationError(
                        f"Checklist item {i} is too long (max 500 characters)",
                        details={"item_index": i, "length": len(item)},
                        suggestions=["Keep checklist items concise and specific"]
                    )
    
    @staticmethod
    def validate_quality_threshold(threshold: float) -> None:
        """Validate quality threshold."""
        if not isinstance(threshold, (int, float)):
            raise ValidationError(
                "Quality threshold must be a number",
                details={"provided_type": type(threshold).__name__, "value": threshold},
                suggestions=["Use a numeric value between 1.0 and 5.0"]
            )
        
        if threshold < 1.0 or threshold > 5.0:
            raise ValidationError(
                "Quality threshold must be between 1.0 and 5.0",
                details={"value": threshold},
                suggestions=["Use a threshold between 1.0 (low) and 5.0 (high)"]
            )
    
    @staticmethod
    def validate_file_upload(file_content: bytes, filename: str, allowed_extensions: List[str]) -> None:
        """Validate uploaded file."""
        if not file_content:
            raise ValidationError(
                "File is empty",
                details={"filename": filename},
                suggestions=["Upload a file with content"]
            )
        
        # Check file extension
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        if file_extension not in allowed_extensions:
            raise ValidationError(
                f"File type not allowed: .{file_extension}",
                details={
                    "filename": filename,
                    "extension": file_extension,
                    "allowed_extensions": allowed_extensions
                },
                suggestions=[f"Use files with extensions: {', '.join(allowed_extensions)}"]
            )
        
        # Check file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(file_content) > max_size:
            raise ValidationError(
                f"File too large: {len(file_content)} bytes",
                details={
                    "filename": filename,
                    "size": len(file_content),
                    "max_size": max_size
                },
                suggestions=["Use a file smaller than 50MB"]
            )
        
        # Basic content validation for .docx files
        if file_extension == 'docx':
            # Check for ZIP file signature (docx files are ZIP archives)
            if not file_content.startswith(b'PK'):
                raise ValidationError(
                    "Invalid .docx file format",
                    details={"filename": filename},
                    suggestions=["Ensure the file is a valid .docx document"]
                )


class RateLimitValidator:
    """Validates and enforces rate limits."""
    
    def __init__(self):
        self.request_counts: Dict[str, Dict[str, int]] = {}
    
    def check_rate_limit(self, client_id: str, endpoint: str, limit: int, window_seconds: int = 3600) -> None:
        """Check if client has exceeded rate limit."""
        import time
        
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        if client_id not in self.request_counts:
            self.request_counts[client_id] = {}
        
        if endpoint not in self.request_counts[client_id]:
            self.request_counts[client_id][endpoint] = 0
        
        # Clean old requests outside the window
        # (Simplified implementation - in production, use Redis or similar)
        
        # Increment current request count
        self.request_counts[client_id][endpoint] += 1
        
        if self.request_counts[client_id][endpoint] > limit:
            raise ValidationError(
                f"Rate limit exceeded for {endpoint}",
                details={
                    "client_id": client_id,
                    "endpoint": endpoint,
                    "current_count": self.request_counts[client_id][endpoint],
                    "limit": limit,
                    "window_seconds": window_seconds
                },
                suggestions=[
                    f"Wait before making more requests to {endpoint}",
                    "Consider batching multiple operations",
                    "Contact support for higher rate limits if needed"
                ]
            )


# Global error handler instance
error_handler = ErrorHandler()
input_validator = InputValidator()
rate_limiter = RateLimitValidator()


def handle_exceptions(func):
    """Decorator for handling exceptions in API endpoints."""
    from functools import wraps
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            error_detail = error_handler.handle_error(e)
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail={
                    "error": error_detail.message,
                    "error_type": error_detail.error_type.value,
                    "details": error_detail.details,
                    "suggestions": error_detail.suggestions
                }
            )
        except ProcessingError as e:
            error_detail = error_handler.handle_error(e)
            from fastapi import HTTPException
            raise HTTPException(
                status_code=422,
                detail={
                    "error": error_detail.message,
                    "error_type": error_detail.error_type.value,
                    "details": error_detail.details,
                    "suggestions": error_detail.suggestions
                }
            )
        except Exception as e:
            error_detail = error_handler.handle_error(e)
            from fastapi import HTTPException
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal server error",
                    "error_type": error_detail.error_type.value,
                    "message": str(e) if hasattr(e, 'message') else str(e),
                    "suggestions": error_detail.suggestions
                }
            )
    
    return wrapper