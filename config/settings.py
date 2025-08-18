"""Configuration settings for the Lawyer Contract Creation System."""

import os
import warnings
from pydantic_settings import BaseSettings
from typing import Optional, List

# Suppress Pydantic model_ namespace warnings
warnings.filterwarnings("ignore", message=".*Field.*has conflict with protected namespace.*")


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    app_name: str = "Lawyer Contract Creation System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-5"
    openai_temperature: float = 0.1  # Low temperature for consistency
    
    # Quality Thresholds
    min_bleu_score: float = 0.8
    min_rouge_average: float = 0.85
    min_meteor_score: float = 0.9
    min_comet_score: float = 0.8
    min_llm_judge_score: float = 4.5
    max_redundancy_score: float = 0.1
    min_completeness_score: float = 0.98
    
    # Generation Settings
    max_regeneration_attempts: int = 3
    quality_gate_enabled: bool = True
    
    # Iteration Settings
    max_completion_iterations: int = 3
    max_refinement_iterations: int = 5
    
    # MLflow Configuration
    mlflow_tracking_uri: str = "file:./mlflow_experiments"
    mlflow_experiment_name: str = "contract_generation"
    
    # File Paths
    data_dir: str = "data"
    skeletons_dir: str = "data/skeletons"
    generated_dir: str = "data/generated"
    references_dir: str = "data/references"
    
    # Database Configuration (for future use)
    database_url: str = "sqlite:///contract_system.db"
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "CONTRACT_",
        "protected_namespaces": (),  # Disable protected namespace warnings
        "extra": "ignore",  # Ignore extra fields
        "arbitrary_types_allowed": True
    }


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


# Global settings instance
settings = get_settings()
