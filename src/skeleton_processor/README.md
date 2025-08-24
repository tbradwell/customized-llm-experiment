# Skeleton Processor Algorithm

## Overview

The Skeleton Processor Algorithm creates document templates by clustering similar paragraphs from multiple documents. It generates skeleton templates with alternative paragraph options and Hebrew RTL delimiters for template-based document creation.

## Algorithm Parameters

- **High homogeneity threshold**: ≥ 0.8
- **Low homogeneity threshold**: < 0.8  
- **Clustering algorithm**: Spherical K-means where K = average number of paragraphs across all documents
- **Distance metric**: Cosine similarity
- **Minimum cluster size**: 1/4 number of documents
- **OpenAI embedding model**: text-embedding-3-large
- **Database**: PostgreSQL with pgvector extension for vector storage and similarity search

## Complete Algorithm Steps

### Phase 1: Document Processing and Embedding Creation (Steps 1-1.2.6)

#### 1. Get all previous documents

#### 1.1 For each document in documents:
- Save document to vector db: document name, document id

#### 1.2 For each paragraph in document:


**1.2.1** Create an embedding of the paragraph with OpenAI.

**1.2.2** Remove all named entities and money amounts and replace them with a placeholder. 

**1.2.3** Save in vector db: 
- a. paragraph embedding 
- b. original paragraph text 
- c. clean paragraph text (after 1.2.2) 
- d. paragraph absolute position 
- e. paragraph relative position 
- f. document id 
- g. save font style for each run in the paragraph

### Phase 2: Clustering and Homogeneity Analysis (Steps 2-3)

#### 2. Cluster the paragraphs using the embedding. Update the paragraph embedding table with the cluster id.

#### 3. For each cluster in the cluster table add the cluster homogeneity score.

### Phase 3: Skeleton Generation (Steps 4-9)

#### 4. Take a random document.

#### 5. For each paragraph in document:

**5.1** Take the clean version from db.

**5.2** Take the embedding from the vector db.

**5.3** Find the closest cluster to the embedded paragraph.

**5.4** Take the medoid (the member closest to that centroid) of the cluster.

**5.5** Take another 2 farthest members from the centroid that are distinct from each other.

**5.6** If the homogeneity score > 0.9 then:
Take the medoid's original text only. Store it in structure that maps paragraph to what it will be swapped to (stored in memory - paragraph_to_replace and db - paragraph_to_replace table). If the homogeneity score <= 0.9 then:
Add both the medoid and the 2 farthest members clean text to a structure that maps paragraph to what it will be swapped to (stored in memory - paragraph_to_replace and db - paragraph_to_replace table).

**5.7** If the current paragraph is mapped to the same cluster as the previous one then assign block f'block_{cluster_id}' otherwise None (add to db and paragraph_to_replace)

**5.8** If the paragraph belongs to a cluster with high homogeneity score then assign it type 'certain' (add to db and paragraph_to_replace)

**5.9** If the paragraph belongs to a cluster with low homogeneity score then assign it type 'uncertain' (add to db and paragraph_to_replace)

#### 6. Initialize empty set: used_cluster_ids = set()

#### 7. Initialize previous_paragraph = None

#### 8. For each member in the paragraph_to_replace:

**8.1** Clean the text in the original document paragraph (the one to be replaced)

**8.2** If previous_paragraph.is_block and (paragraph.is_block is False or previous_paragraph.cluster_id != paragraph.cluster_id) then:
- Add '{%' (closing delimiter)

**8.3** If paragraph.is_block and the paragraph.cluster_id not in used_cluster_ids then: 
- Add wrapping delimiter '%}' (opening delimiter) and add cluster_id to used_cluster_ids

**8.4** If paragraph.type is 'uncertain' and (paragraph.is_block False or paragraph.block_id not in used_cluster_ids) then:
- Add '~}' delimiter to document

**8.5** Add to the document where the text was cleaned member of the medoid and 2 farthest, each in a different paragraph one under the other with their corresponding font style.

**8.6** If paragraph is block then:
- Set previous_paragraph = current paragraph

#### 9. If last paragraph was block (not None) then:
- Add '{%' (closing delimiter)

## Hebrew RTL Delimiters

The algorithm uses Hebrew Right-to-Left delimiters:

- **`%}`** - Block start delimiter (opening)
- **`{%`** - Block end delimiter (closing)  
- **`~}`** - Uncertain content delimiter

## Implementation Architecture

### Core Components

#### 1. SkeletonProcessor (Main Orchestrator)
```python
from skeleton_processor import SkeletonProcessor

processor = SkeletonProcessor(
    openai_api_key="your-api-key",
    embedding_model="text-embedding-3-large",
    high_homogeneity_threshold=0.8,
    low_homogeneity_threshold=0.8,
    language="he"
)

skeleton_doc = processor.process_documents_to_skeleton(document_paths)
```

#### 2. DocumentPreprocessor (Steps 1.1-1.2.6)
- Document loading and paragraph extraction
- Named entity and number replacement
- OpenAI embedding creation with position encoding
- Font style preservation

#### 3. ClusteringEngine (Steps 2-3)
- Spherical K-means clustering with cosine similarity
- Homogeneity score calculation
- Representative member identification
- Certainty type assignment

#### 4. SkeletonGenerator (Steps 4-9)
- Random document selection
- Paragraph replacement structure creation
- Hebrew RTL delimiter insertion
- Template document generation

### Data Models

#### Paragraph Model
```python
@dataclass
class Paragraph:
    id: str
    document_id: str
    original_text: str
    clean_text: str
    absolute_position: int
    relative_position: float
    embedding: np.ndarray
    cluster_id: Optional[int]
    block_assignment: Optional[str]
    certainty_type: str  # "certain" or "uncertain"
    font_style: Dict[str, Any]
    is_block: bool
```

#### Cluster Model
```python
@dataclass
class Cluster:
    id: int
    paragraph_ids: List[str]
    centroid: np.ndarray
    homogeneity_score: float
    medoid_id: str
    farthest_member_ids: List[str]
```

#### SkeletonDocument Model
```python
@dataclass
class SkeletonDocument:
    id: str
    source_document_ids: List[str]
    template_path: str
    content_blocks: Dict[int, List[str]]
    delimiter_positions: Dict[int, str]
    total_paragraphs_processed: int
    total_clusters_found: int
```

### Utility Classes

#### EmbeddingClient
- OpenAI API integration with batching
- Position embedding creation
- Embedding concatenation
- Error handling and retry logic

#### TextCleaner
- Named entity recognition with spaCy
- Number pattern detection and replacement
- Hebrew and English support
- Placeholder management

#### SphericalKMeans
- Custom clustering for high-dimensional embeddings
- Cosine similarity distance metric
- K-means++ initialization
- Convergence detection

#### DelimiterFormatter
- Hebrew RTL delimiter management
- Block tracking and validation
- Document formatting

## Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export EMBEDDING_MODEL="text-embedding-3-large"
export HIGH_HOMOGENEITY_THRESHOLD="0.8"
export LOW_HOMOGENEITY_THRESHOLD="0.8"
export LANGUAGE="he"
export DATABASE_URL="postgresql://user:pass@localhost:5433/legal_ai_db"
```

### Configuration File (JSON)
```json
{
  "openai_api_key": "your-openai-api-key",
  "embedding_model": "text-embedding-3-large",
  "high_homogeneity_threshold": 0.8,
  "low_homogeneity_threshold": 0.8,
  "language": "he",
  "batch_size": 100,
  "database_url": "postgresql://user:pass@localhost:5432/skeleton_db",
  "output_directory": "skeleton_outputs"
}
```

## API Usage

### REST Endpoints

#### Upload Documents
```bash
POST /skeleton/upload-documents
Content-Type: multipart/form-data

# Upload multiple DOCX files
curl -X POST -F "files=@contract1.docx" -F "files=@contract2.docx" \
  http://localhost:8000/skeleton/upload-documents
```

#### Process Documents
```bash
POST /skeleton/process
Content-Type: application/json

{
  "upload_id": "123e4567-e89b-12d3-a456-426614174000",
  "async_processing": true
}
```

#### Check Processing Status
```bash
GET /skeleton/status/{job_id}

# Response:
{
  "job_id": "456e7890-e89b-12d3-a456-426614174000",
  "status": "completed",
  "progress": 100,
  "skeleton_id": "789e0123-e89b-12d3-a456-426614174000",
  "template_download_url": "/skeleton/download/789e0123-e89b-12d3-a456-426614174000"
}
```

#### Download Template
```bash
GET /skeleton/download/{skeleton_id}
# Returns DOCX file with skeleton template
```

#### Get Statistics
```bash
GET /skeleton/stats/{job_id}

# Response includes:
{
  "input_statistics": {
    "document_count": 3,
    "total_paragraphs": 87,
    "total_clusters": 12
  },
  "clustering_statistics": {
    "clusters_created": 12,
    "blocks_generated": 8,
    "uncertain_blocks": 4
  },
  "algorithm_parameters": {
    "embedding_model": "text-embedding-3-large",
    "clustering_algorithm": "Spherical K-means"
  }
}
```

## Example Usage

### Python Code Example
```python
from skeleton_processor import SkeletonProcessor
from skeleton_processor.config import SkeletonProcessorConfig

# Load configuration
config = SkeletonProcessorConfig.from_environment()

# Initialize processor
processor = SkeletonProcessor(
    openai_api_key=config.openai_api_key,
    embedding_model=config.embedding_model,
    high_homogeneity_threshold=config.high_homogeneity_threshold,
    low_homogeneity_threshold=config.low_homogeneity_threshold,
    language=config.language
)

# Process documents
document_paths = [
    "contracts/contract1.docx",
    "contracts/contract2.docx", 
    "contracts/contract3.docx"
]

# Validate inputs
is_valid, issues = processor.validate_input_documents(document_paths)
if not is_valid:
    print(f"Validation issues: {issues}")
    exit(1)

# Generate skeleton
skeleton_doc = processor.process_documents_to_skeleton(document_paths)

print(f"Generated skeleton: {skeleton_doc.id}")
print(f"Template path: {skeleton_doc.template_path}")
print(f"Processed {skeleton_doc.total_paragraphs_processed} paragraphs")
print(f"Created {skeleton_doc.total_clusters_found} clusters")
```

### Output Example

The generated skeleton template will contain:

```
Original paragraph content here.

%}
Alternative paragraph option 1 (medoid)

Alternative paragraph option 2 (farthest member 1)

Alternative paragraph option 3 (farthest member 2)
{%

More original content.

~}
Uncertain content with single alternative
```

## Testing

### Run Tests
```bash
cd src/skeleton_processor
python test_skeleton_processor.py
```

### Test Coverage
- Unit tests for all core components
- Integration tests for full pipeline
- API endpoint testing
- Configuration validation
- Model serialization/deserialization

## Performance Considerations

### Scalability
- **Batch Processing**: OpenAI API calls batched for efficiency
- **Memory Management**: Streaming for large document sets
- **Clustering Optimization**: Approximate methods for >10k paragraphs
- **Caching**: Embedding caching to avoid re-computation

### Optimization Tips
- Use smaller embedding models for faster processing
- Adjust batch sizes based on available memory
- Consider parallel processing for multiple document sets
- Implement embedding caching for repeated documents

## Error Handling

### Common Issues and Solutions

#### OpenAI API Errors
- **Rate limits**: Automatic retry with exponential backoff
- **Authentication**: Validate API key in configuration
- **Model availability**: Fallback to alternative models

#### Document Processing Errors
- **Corrupted DOCX**: Validation before processing
- **Empty documents**: Skip with warning
- **Large files**: Size limits and chunking

#### Clustering Errors
- **Insufficient data**: Minimum document requirements
- **Memory issues**: Batch processing and streaming
- **Convergence failures**: Alternative initialization

## Dependencies

### Required Packages
```
openai>=1.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
spacy>=3.4.0
python-docx>=0.8.11
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
tenacity>=8.0.0
psycopg2-binary>=2.9.0  # For PostgreSQL
```

### Language Models
```bash
# Install Hebrew spaCy model
python -m spacy download he_core_news_sm

# Install English spaCy model  
python -m spacy download en_core_web_sm
```

## Database Schema (PostgreSQL + pgvector)

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Paragraphs table with vector embeddings
CREATE TABLE paragraphs (
    id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES documents(id),
    original_text TEXT NOT NULL,
    clean_text TEXT NOT NULL,
    absolute_position INTEGER NOT NULL,
    relative_position FLOAT NOT NULL,
    embedding vector(3072),  -- text-embedding-3-large dimension
    cluster_id INTEGER,
    block_assignment TEXT,
    certainty_type TEXT,
    font_style JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Clusters table
CREATE TABLE clusters (
    id INTEGER PRIMARY KEY,
    centroid vector(3072),
    homogeneity_score FLOAT NOT NULL,
    medoid_id TEXT,
    farthest_member_ids TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Skeleton documents table
CREATE TABLE skeleton_documents (
    id TEXT PRIMARY KEY,
    source_document_ids TEXT[],
    template_path TEXT NOT NULL,
    content_blocks JSONB,
    delimiter_positions JSONB,
    total_paragraphs_processed INTEGER,
    total_clusters_found INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_paragraphs_embedding ON paragraphs USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_paragraphs_cluster ON paragraphs(cluster_id);
CREATE INDEX idx_paragraphs_document ON paragraphs(document_id);
```

## Deployment

### Docker Configuration
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install spaCy models
RUN python -m spacy download he_core_news_sm
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app
ENV OPENAI_API_KEY=""

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Setup
```bash
# Development setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install language models
python -m spacy download he_core_news_sm
python -m spacy download en_core_web_sm

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export DATABASE_URL="postgresql://user:pass@localhost:5432/skeleton_db"

# Run application
python -m uvicorn src.api.main:app --reload
```

## License

This implementation is part of the Customized LLM Experiments project.

## Contributing

1. Follow the existing code structure and patterns
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure compatibility with Hebrew RTL formatting
5. Validate against the algorithm specification

## Support

For issues and questions:
1. Check the test suite for usage examples
2. Review the API documentation at `/docs` endpoint
3. Validate configuration with the health check endpoint
4. Check logs for detailed error information
