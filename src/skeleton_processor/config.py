"""Configuration management for skeleton processor."""

import os
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SkeletonProcessorConfig:
    """Configuration for skeleton processor algorithm."""
    
    # OpenAI Configuration
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    
    # Algorithm Parameters
    high_homogeneity_threshold: float = 0.8
    low_homogeneity_threshold: float = 0.8
    min_cluster_size_ratio: float = 0.25
    
    # Language and Processing
    language: str = "he"  # Hebrew by default
    batch_size: int = 100
    
    # Database Configuration (for future use)
    database_url: Optional[str] = None
    pgvector_dimension: int = 3072  # text-embedding-3-large dimension
    
    # File Processing
    max_file_size_mb: int = 50
    supported_formats: tuple = (".docx",)
    
    # Output Configuration
    output_directory: str = "skeleton_outputs"
    template_prefix: str = "skeleton_template"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_environment(cls) -> 'SkeletonProcessorConfig':
        """Create configuration from environment variables."""
        
        # Required environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Optional environment variables with defaults
        config = cls(
            openai_api_key=openai_api_key,
            embedding_model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large'),
            high_homogeneity_threshold=float(os.getenv('HIGH_HOMOGENEITY_THRESHOLD', '0.8')),
            low_homogeneity_threshold=float(os.getenv('LOW_HOMOGENEITY_THRESHOLD', '0.8')),
            min_cluster_size_ratio=float(os.getenv('MIN_CLUSTER_SIZE_RATIO', '0.25')),
            language=os.getenv('LANGUAGE', 'he'),
            batch_size=int(os.getenv('BATCH_SIZE', '100')),
            database_url=os.getenv('DATABASE_URL'),
            pgvector_dimension=int(os.getenv('PGVECTOR_DIMENSION', '3072')),
            max_file_size_mb=int(os.getenv('MAX_FILE_SIZE_MB', '50')),
            output_directory=os.getenv('OUTPUT_DIRECTORY', 'skeleton_outputs'),
            template_prefix=os.getenv('TEMPLATE_PREFIX', 'skeleton_template'),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE')
        )
        
        logger.info("Configuration loaded from environment")
        return config
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SkeletonProcessorConfig':
        """Create configuration from dictionary."""
        
        # Validate required fields
        if 'openai_api_key' not in config_dict:
            raise ValueError("openai_api_key is required in configuration")
        
        # Create config with provided values, using defaults for missing ones
        config = cls(**config_dict)
        
        logger.info("Configuration loaded from dictionary")
        return config
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration values."""
        issues = []
        
        # Validate API key
        if not self.openai_api_key:
            issues.append("OpenAI API key is required")
        
        # Validate thresholds
        if not 0 <= self.high_homogeneity_threshold <= 1:
            issues.append("High homogeneity threshold must be between 0 and 1")
        
        if not 0 <= self.low_homogeneity_threshold <= 1:
            issues.append("Low homogeneity threshold must be between 0 and 1")
        
        # Validate embedding model
        valid_models = [
            "text-embedding-3-large",
            "text-embedding-3-small", 
            "text-embedding-ada-002"
        ]
        if self.embedding_model not in valid_models:
            issues.append(f"Embedding model must be one of: {valid_models}")
        
        # Validate language
        valid_languages = ["he", "en"]
        if self.language not in valid_languages:
            issues.append(f"Language must be one of: {valid_languages}")
        
        # Validate batch size
        if self.batch_size <= 0:
            issues.append("Batch size must be positive")
        
        # Validate file size
        if self.max_file_size_mb <= 0:
            issues.append("Max file size must be positive")
        
        # Validate output directory
        if not self.output_directory:
            issues.append("Output directory cannot be empty")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Configuration validation passed")
        else:
            logger.warning(f"Configuration validation failed: {issues}")
        
        return is_valid, issues
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        
        # Configure logging level
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        # Configure logging format
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure handlers
        handlers = [logging.StreamHandler()]  # Always log to console
        
        if self.log_file:
            handlers.append(logging.FileHandler(self.log_file))
        
        # Setup basic config
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers,
            force=True  # Override any existing configuration
        )
        
        logger.info(f"Logging configured: level={self.log_level}, file={self.log_file}")
    
    def setup_output_directory(self) -> str:
        """Setup output directory for skeleton files."""
        
        os.makedirs(self.output_directory, exist_ok=True)
        
        abs_path = os.path.abspath(self.output_directory)
        logger.info(f"Output directory ready: {abs_path}")
        
        return abs_path
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for the configured model."""
        
        model_dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        
        return model_dimensions.get(self.embedding_model, 1536)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        
        return {
            'openai_api_key': '***HIDDEN***',  # Don't expose API key
            'embedding_model': self.embedding_model,
            'high_homogeneity_threshold': self.high_homogeneity_threshold,
            'low_homogeneity_threshold': self.low_homogeneity_threshold,
            'min_cluster_size_ratio': self.min_cluster_size_ratio,
            'language': self.language,
            'batch_size': self.batch_size,
            'database_url': '***HIDDEN***' if self.database_url else None,
            'pgvector_dimension': self.pgvector_dimension,
            'max_file_size_mb': self.max_file_size_mb,
            'supported_formats': self.supported_formats,
            'output_directory': self.output_directory,
            'template_prefix': self.template_prefix,
            'log_level': self.log_level,
            'log_file': self.log_file
        }
    
    def __str__(self) -> str:
        """String representation of configuration (safe for logging)."""
        
        config_dict = self.to_dict()
        return f"SkeletonProcessorConfig({config_dict})"


def load_config_from_file(file_path: str) -> SkeletonProcessorConfig:
    """Load configuration from JSON or YAML file."""
    
    import json
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                config_dict = json.load(f)
            elif file_path.endswith(('.yml', '.yaml')):
                try:
                    import yaml
                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required to load YAML configuration files")
            else:
                raise ValueError("Configuration file must be JSON or YAML")
        
        return SkeletonProcessorConfig.from_dict(config_dict)
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {file_path}: {e}")
        raise


def create_default_config_file(file_path: str) -> None:
    """Create a default configuration file."""
    
    import json
    
    default_config = {
        "openai_api_key": "your-openai-api-key-here",
        "embedding_model": "text-embedding-3-large",
        "high_homogeneity_threshold": 0.8,
        "low_homogeneity_threshold": 0.8,
        "min_cluster_size_ratio": 0.25,
        "language": "he",
        "batch_size": 100,
        "database_url": "postgresql://user:password@localhost:5432/skeleton_db",
        "pgvector_dimension": 3072,
        "max_file_size_mb": 50,
        "output_directory": "skeleton_outputs",
        "template_prefix": "skeleton_template",
        "log_level": "INFO",
        "log_file": "skeleton_processor.log"
    }
    
    with open(file_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    logger.info(f"Default configuration file created: {file_path}")


# Global configuration instance
_global_config: Optional[SkeletonProcessorConfig] = None


def get_config() -> SkeletonProcessorConfig:
    """Get global configuration instance."""
    
    global _global_config
    
    if _global_config is None:
        raise RuntimeError("Configuration not initialized. Call set_config() first.")
    
    return _global_config


def set_config(config: SkeletonProcessorConfig) -> None:
    """Set global configuration instance."""
    
    global _global_config
    
    # Validate configuration
    is_valid, issues = config.validate()
    if not is_valid:
        raise ValueError(f"Invalid configuration: {issues}")
    
    _global_config = config
    
    # Setup logging and output directory
    config.setup_logging()
    config.setup_output_directory()
    
    logger.info("Global configuration set successfully")


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    
    global _global_config
    _global_config = None
    
    logger.info("Global configuration reset")
