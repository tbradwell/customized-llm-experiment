# Lawyer Contract Creation System

A quality-focused AI-powered system for generating high-quality legal contracts using intelligent document processing and comprehensive quality evaluation metrics.

## Overview

The Lawyer Contract Creation System is designed according to the PRD specifications to prioritize **quality and accuracy over speed**. It features:

- **Multi-metric Quality Evaluation**: BLEU, ROUGE, METEOR, COMET, LLM Judge, redundancy detection, and completeness scoring
- **Quality Gates Pipeline**: Iterative generation with quality assurance checkpoints
- **Comprehensive Document Processing**: Advanced .docx handling with placeholder replacement
- **MLflow Integration**: Complete experiment tracking and model performance monitoring
- **Professional API**: RESTful API with comprehensive documentation

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository (if not already done)
# cd cutomizd-LLM-experiments

# Install dependencies
pip install -r requirements.txt

# Setup environment and download models
python setup_environment.py

# Copy environment configuration
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Configuration

Edit `.env` file and set your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the Server

```bash
# Using the startup script (recommended)
python start_server.py

# Or directly with uvicorn
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Architecture

### Core Components

1. **Document Processor** (`src/core/document_processor.py`)
   - Handles .docx file manipulation and copying
   - Finds and replaces placeholders with intelligent mapping
   - Validates document structure and completeness

2. **Content Generator** (`src/core/content_generator.py`)
   - OpenAI-powered intelligent content generation
   - Context-aware contract clause creation
   - Professional legal language optimization

3. **Quality Pipeline** (`src/core/quality_pipeline.py`)
   - Multi-stage quality assurance with iterative improvement
   - Comprehensive quality gates enforcement
   - Quality score aggregation and threshold management

4. **Evaluation Framework** (`src/evaluation/`)
   - **Metrics Calculator**: BLEU, ROUGE, METEOR, redundancy, completeness
   - **COMET Evaluator**: Neural-based semantic quality assessment
   - **LLM Judge**: AI-powered holistic quality evaluation

5. **MLflow Tracker** (`src/utils/mlflow_tracker.py`)
   - Experiment tracking and model performance monitoring
   - Quality metrics logging and comparison
   - Artifact management for contracts and reports

### Quality Metrics

The system implements all PRD-specified evaluation metrics:

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| BLEU Score | > 0.8 | Precision against reference contracts |
| ROUGE Average | > 0.85 | Content coverage and recall |
| METEOR Score | > 0.9 | Semantic similarity with synonyms |
| COMET Score | > 0.8 | Neural-based quality assessment |
| LLM Judge | > 4.5/5 | Holistic AI evaluation |
| Redundancy | < 0.1 | Unnecessary repetition detection |
| Completeness | > 98% | Required elements presence |

## API Usage

### 1. Upload Contract Skeleton

```bash
curl -X POST "http://localhost:8000/upload/skeleton" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@examples/service_agreement_skeleton.docx"
```

### 2. Generate Contract

```bash
curl -X POST "http://localhost:8000/contracts/generate?skeleton_id=SKELETON_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "contract_data": {
      "client_name": "Acme Corporation",
      "provider_name": "Expert Legal Services",
      "contract_value": "$75,000",
      "start_date": "2024-03-01",
      "service_description": "Comprehensive legal consultation services"
    },
    "checklist": ["Include payment terms", "Add confidentiality clause"],
    "quality_threshold": 4.5
  }'
```

### 3. Download Generated Contract

```bash
curl -X GET "http://localhost:8000/contracts/download/CONTRACT_ID" \
  --output generated_contract.docx
```

### 4. Get Quality Report

```bash
curl -X GET "http://localhost:8000/contracts/quality-report/CONTRACT_ID"
```

## Running Experiments

### Command Line Experiments

The system provides several ways to run experiments for testing and evaluation:

#### 1. Configurable Experiment Runner

```bash
# Run experiment with default settings
python run_configurable_experiment.py

# This will run 3 experiments:
# - Default settings (from config/settings.py)
# - Enhanced iterations (5 completion, 10 refinement)
# - Minimal iterations (1 completion, 1 refinement)
```

#### 2. Quality Pipeline Experiment

```bash
# Run full quality pipeline with MLflow tracking
python -c "
from src.core.quality_pipeline import QualityAssurancePipeline
from src.core.document_processor import DocumentProcessor
from docx import Document

# Initialize pipeline
pipeline = QualityAssurancePipeline(enable_mlflow=True)

# Load ground truth for evaluation
doc_processor = DocumentProcessor()
gt_doc = Document('examples/amit_test/gt.docx')
ground_truth = doc_processor.extract_text_content(gt_doc)

# Contract data
contract_data = {
    'contract_type': 'legal_claim',
    'client_name': 'Client Name',
    'provider_name': 'Provider Name',
    'source_content': 'Contract source content'
}

# Run pipeline with evaluation
result = pipeline.process_contract(
    skeleton_path='data/skeletons/skeleton_oracle.docx',
    contract_data=contract_data,
    reference_contracts=[ground_truth],
    output_path='outputs/experiment_result.docx',
    experiment_name='my_experiment'
)

print(f'Success: {result.success}')
print(f'Quality scores: {result.quality_scores}')
"
```

#### 3. Simple Generation Test

```bash
# Test basic generation without full pipeline
python -c "
from src.core.content_generator import IntelligentContentGenerator, GenerationContext
from src.core.document_processor import DocumentProcessor
from docx import Document

# Initialize
generator = IntelligentContentGenerator()
doc_processor = DocumentProcessor()

# Load skeleton
skeleton_doc = Document('data/skeletons/skeleton_oracle.docx')
skeleton_text = doc_processor.extract_text_content(skeleton_doc)
placeholders = [p.field_name for p in doc_processor.find_placeholders(skeleton_doc)]

# Create context
context = GenerationContext(
    contract_type='service_agreement',
    skeleton_text=skeleton_text,
    placeholders=placeholders,
    contract_data={'client_name': 'Test Client', 'provider_name': 'Test Provider'}
)

# Generate
result = generator.generate_complete_contract(context)
print(f'Success: {result.success}')
print(f'Content length: {len(result.generated_content)}')
"
```

#### 4. Custom Experiment with Specific Data

```bash
# Example: Run experiment with specific test case
python -c "
import sys
sys.path.append('.')
from src.core.quality_pipeline import QualityAssurancePipeline

pipeline = QualityAssurancePipeline(enable_mlflow=True)

# Custom contract data for your specific case
contract_data = {
    'contract_type': 'legal_claim',
    'client_name': 'זוהי קריכלי',
    'provider_name': 'חברת הביטוח', 
    'case_type': 'נזקי רטיבות',
    'location': 'חיים הרצוג 7, הוד השרון'
}

result = pipeline.process_contract(
    skeleton_path='data/skeletons/skeleton_oracle.docx',
    contract_data=contract_data,
    output_path='outputs/custom_experiment.docx',
    experiment_name='custom_test_experiment'
)
"
```

### MLflow Experiment Tracking

All experiments are automatically tracked in MLflow:

```bash
# Start MLflow UI to view experiments
mlflow ui

# Then open http://localhost:5000 in your browser
```

**What gets tracked:**
- Quality metrics (BLEU, ROUGE, METEOR, COMET, LLM Judge, Redundancy, Completeness)
- Generation metadata (iterations, time, placeholders)
- Contract artifacts (generated documents, skeletons, reports)
- Code artifacts (for reproducibility)
- Parameters and tags

### Experiment Configuration

You can customize experiments by:

1. **Environment Variables:**
```bash
export CONTRACT_MAX_COMPLETION_ITERATIONS=5
export CONTRACT_MAX_REFINEMENT_ITERATIONS=10
export CONTRACT_MIN_BLEU_SCORE=0.7
```

2. **Direct Parameters:**
```python
# Override in generation call
result = generator.generate_complete_contract(
    context,
    max_completion_iterations=5,
    max_refinement_iterations=10
)
```

3. **Settings File:**
Edit `config/settings.py` for default values

### Evaluation Against Ground Truth

To evaluate against existing contracts:

```bash
# Using the evaluation framework
python -c "
from src.evaluation.metrics import MetricsCalculator
from src.core.document_processor import DocumentProcessor
from docx import Document

# Load documents
doc_processor = DocumentProcessor()
generated_doc = Document('outputs/generated_contract.docx')
reference_doc = Document('examples/amit_test/gt.docx')

generated_text = doc_processor.extract_text_content(generated_doc)
reference_text = doc_processor.extract_text_content(reference_doc)

# Calculate metrics
metrics_calc = MetricsCalculator()
bleu_result = metrics_calc.calculate_bleu_score(generated_text, [reference_text])
rouge_result = metrics_calc.calculate_rouge_scores(generated_text, [reference_text])

print(f'BLEU: {bleu_result.score:.3f}')
print(f'ROUGE: {rouge_result.score:.3f}')
"
```

## Examples

### Example Contract Data

```json
{
  "contract_type": "service_agreement",
  "client_name": "Innovative Tech Solutions Ltd.",
  "client_address": "789 Innovation Drive, Tech Valley, NY 12180",
  "provider_name": "Professional Legal Advisors LLP",
  "provider_address": "321 Legal Center, Suite 450, Metropolitan City, NY 10001",
  "contract_value": "$125,000",
  "start_date": "2024-03-01",
  "end_date": "2025-02-28",
  "service_description": "Comprehensive legal support for technology commercialization and IP protection",
  "payment_terms": "Monthly retainer of $10,417, payable on the first business day of each month",
  "confidentiality_clause": "All proprietary information shall be maintained in strict confidence for seven (7) years"
}
```

### Python Client Example

```python
import requests
import json

# Upload skeleton
with open('skeleton.docx', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload/skeleton',
        files={'file': f}
    )
skeleton_id = response.json()['skeleton_id']

# Generate contract
contract_data = {
    "contract_data": {
        "client_name": "Test Client Corp",
        "provider_name": "Test Provider LLC",
        "contract_value": "$50,000"
    },
    "quality_threshold": 4.5
}

response = requests.post(
    f'http://localhost:8000/contracts/generate?skeleton_id={skeleton_id}',
    json=contract_data
)

result = response.json()
print(f"Quality Score: {result['quality_score']}")
print(f"Contract ID: {result['contract_id']}")
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Only integration tests
pytest tests/test_metrics.py  # Specific test file

# Run with coverage
pytest --cov=src --cov-report=html
```

### Development Mode

```bash
# Start with auto-reload for development
python start_server.py --reload

# Or with uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Creating Sample Documents

```bash
# Create sample .docx files from text templates
python examples/create_sample_docx.py
```

## File Structure

```
cutomizd-LLM-experiments/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # Main API endpoints
│   │   └── models.py          # Pydantic models
│   ├── core/                  # Core business logic
│   │   ├── document_processor.py
│   │   ├── content_generator.py
│   │   └── quality_pipeline.py
│   ├── evaluation/            # Quality evaluation framework
│   │   ├── metrics.py         # BLEU, ROUGE, METEOR, etc.
│   │   └── llm_judge.py       # AI-based evaluation
│   └── utils/
│       └── mlflow_tracker.py  # Experiment tracking
├── config/
│   └── settings.py            # Configuration management
├── data/
│   ├── skeletons/             # Contract skeleton templates
│   ├── generated/             # Generated contracts
│   └── references/            # Reference contracts
├── examples/                  # Example data and scripts
├── tests/                     # Comprehensive test suite
├── requirements.txt           # Python dependencies
├── setup_environment.py      # Environment setup script
├── start_server.py           # Server startup script
└── README.md                 # This file
```

## Quality Assurance

The system implements a comprehensive quality assurance pipeline:

1. **Pre-Generation Validation**
   - Skeleton document validation
   - Contract data completeness checking
   - Placeholder mapping verification

2. **Iterative Generation with Quality Gates**
   - Multi-metric evaluation after each generation attempt
   - Automatic regeneration if quality thresholds not met
   - Maximum iteration limits to prevent infinite loops

3. **Post-Generation Assessment**
   - Final quality report generation
   - Comprehensive metric logging to MLflow
   - Quality gate status tracking

## Configuration

### Environment Variables

All configuration is managed through environment variables with the `CONTRACT_` prefix:

```bash
# Required
OPENAI_API_KEY=your_api_key_here

# Optional (with defaults)
CONTRACT_OPENAI_MODEL=gpt-5
CONTRACT_MIN_BLEU_SCORE=0.8
CONTRACT_MIN_ROUGE_AVERAGE=0.85
CONTRACT_MAX_REGENERATION_ATTEMPTS=3
```

### Quality Thresholds

Quality thresholds can be adjusted in the `.env` file or settings:

- `CONTRACT_MIN_BLEU_SCORE`: Minimum BLEU score (default: 0.8)
- `CONTRACT_MIN_ROUGE_AVERAGE`: Minimum ROUGE average (default: 0.85)
- `CONTRACT_MIN_METEOR_SCORE`: Minimum METEOR score (default: 0.9)
- `CONTRACT_MIN_LLM_JUDGE_SCORE`: Minimum LLM judge score (default: 4.5)
- `CONTRACT_MAX_REDUNDANCY_SCORE`: Maximum redundancy (default: 0.1)

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```
   Error: OpenAI API key not configured
   Solution: Set OPENAI_API_KEY in .env file
   ```

2. **spaCy Model Missing**
   ```
   Warning: spaCy model 'en_core_web_sm' not found
   Solution: python -m spacy download en_core_web_sm
   ```

3. **COMET Model Download Issues**
   ```
   Warning: Could not initialize COMET model
   Solution: Check internet connection, model will use fallback scoring
   ```

4. **MLflow Database Issues**
   ```
   Error: MLflow tracking failed
   Solution: Check MLflow configuration in settings.py
   ```

### Performance Optimization

- **Quality over Speed**: The system prioritizes accuracy over generation speed
- **Iteration Limits**: Adjust `max_regeneration_attempts` based on quality requirements
- **Model Selection**: Use `gpt-5` for highest quality
- **Batch Processing**: Use batch endpoints for multiple contracts

## Contributing

1. **Code Style**: Follow PEP 8, use Black formatter
2. **Testing**: Add tests for new features, maintain >80% coverage
3. **Documentation**: Update README and docstrings
4. **Quality Gates**: Ensure all quality metrics pass

## License

This project is part of a custom LLM experiments repository focused on legal document generation with quality assurance.

## Support

For issues and support:
1. Check the troubleshooting section above
2. Review the API documentation at `/docs`
3. Check MLflow tracking for experiment insights
4. Examine test outputs for component validation