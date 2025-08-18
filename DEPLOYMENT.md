# Deployment Guide

This guide covers deployment options for the Lawyer Contract Creation System in various environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Production Deployment](#production-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Configuration Management](#configuration-management)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Security Considerations](#security-considerations)

## Local Development

### Quick Start

```bash
# 1. Setup environment
python setup_environment.py

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your OpenAI API key

# 3. Start development server
python start_server.py --reload
```

### Development Configuration

For development, use these settings in `.env`:

```bash
CONTRACT_DEBUG=true
CONTRACT_OPENAI_TEMPERATURE=0.1
CONTRACT_MAX_REGENERATION_ATTEMPTS=2
CONTRACT_QUALITY_GATE_ENABLED=true
```

## Production Deployment

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores, 2.5GHz
- RAM: 4GB
- Storage: 10GB
- Python: 3.8+
- OS: Linux (Ubuntu 20.04+ recommended)

**Recommended Requirements:**
- CPU: 4 cores, 3.0GHz
- RAM: 8GB
- Storage: 50GB SSD
- Python: 3.10+

### Production Setup

1. **Install System Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3-pip nginx supervisor
   ```

2. **Create Application User**
   ```bash
   sudo useradd -m -s /bin/bash contractapp
   sudo usermod -aG sudo contractapp
   ```

3. **Setup Application**
   ```bash
   sudo -u contractapp bash
   cd /home/contractapp
   
   # Clone/copy application code
   git clone [repository] contract-system
   cd contract-system
   
   # Create virtual environment
   python3.10 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install gunicorn
   
   # Setup environment
   python setup_environment.py
   ```

4. **Configure Environment**
   ```bash
   # Production environment configuration
   cat > .env << EOF
   OPENAI_API_KEY=your_production_api_key
   CONTRACT_DEBUG=false
   CONTRACT_MLFLOW_TRACKING_URI=postgresql://user:pass@localhost:5432/mlflow
   CONTRACT_DATABASE_URL=postgresql://user:pass@localhost:5432/contracts
   EOF
   ```

5. **Configure Gunicorn**
   ```bash
   cat > gunicorn.conf.py << EOF
   bind = "127.0.0.1:8000"
   workers = 4
   worker_class = "uvicorn.workers.UvicornWorker"
   worker_connections = 1000
   max_requests = 1000
   max_requests_jitter = 100
   timeout = 120
   keepalive = 5
   user = "contractapp"
   group = "contractapp"
   preload_app = True
   capture_output = True
   enable_stdio_inheritance = True
   EOF
   ```

6. **Configure Supervisor**
   ```bash
   sudo cat > /etc/supervisor/conf.d/contract-system.conf << EOF
   [program:contract-system]
   command=/home/contractapp/contract-system/venv/bin/gunicorn src.api.main:app -c gunicorn.conf.py
   directory=/home/contractapp/contract-system
   user=contractapp
   group=contractapp
   autostart=true
   autorestart=true
   redirect_stderr=true
   stdout_logfile=/var/log/contract-system.log
   environment=PATH="/home/contractapp/contract-system/venv/bin"
   EOF
   
   sudo supervisorctl reread
   sudo supervisorctl update
   sudo supervisorctl start contract-system
   ```

7. **Configure Nginx**
   ```bash
   sudo cat > /etc/nginx/sites-available/contract-system << EOF
   server {
       listen 80;
       server_name your-domain.com;
       
       client_max_body_size 50M;
       
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host \$host;
           proxy_set_header X-Real-IP \$remote_addr;
           proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto \$scheme;
           proxy_read_timeout 300;
           proxy_connect_timeout 300;
           proxy_send_timeout 300;
       }
       
       location /static/ {
           alias /home/contractapp/contract-system/static/;
           expires 1y;
           add_header Cache-Control "public, immutable";
       }
   }
   EOF
   
   sudo ln -s /etc/nginx/sites-available/contract-system /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CONTRACT_DEBUG=false

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy project
COPY . .

# Create data directories
RUN mkdir -p data/skeletons data/generated data/references

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "start_server.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  contract-system:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CONTRACT_MLFLOW_TRACKING_URI=http://mlflow:5000
      - CONTRACT_DATABASE_URL=postgresql://postgres:password@db:5432/contracts
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - db
      - mlflow
    restart: unless-stopped

  mlflow:
    image: python:3.10-slim
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://postgres:password@db:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    command: >
      bash -c "
        pip install mlflow psycopg2-binary &&
        mlflow server 
          --backend-store-uri postgresql://postgres:password@db:5432/mlflow
          --default-artifact-root /mlflow/artifacts
          --host 0.0.0.0
          --port 5000
      "
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=contracts
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - contract-system
    restart: unless-stopped

volumes:
  postgres_data:
  mlflow_artifacts:
```

### Build and Deploy

```bash
# Build and start services
docker-compose up -d

# Check logs
docker-compose logs -f contract-system

# Scale application
docker-compose up -d --scale contract-system=3

# Update application
docker-compose build contract-system
docker-compose up -d contract-system
```

## Cloud Deployment

### AWS Deployment

#### Using AWS Elastic Container Service (ECS)

1. **Create ECR Repository**
   ```bash
   aws ecr create-repository --repository-name contract-system
   ```

2. **Build and Push Image**
   ```bash
   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

   # Build and tag image
   docker build -t contract-system .
   docker tag contract-system:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/contract-system:latest

   # Push image
   docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/contract-system:latest
   ```

3. **Create ECS Task Definition**
   ```json
   {
     "family": "contract-system",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "contract-system",
         "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/contract-system:latest",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "OPENAI_API_KEY",
             "value": "your-api-key"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/contract-system",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

#### Using AWS Lambda (for lighter workloads)

```python
# lambda_handler.py
import json
from mangum import Mangum
from src.api.main import app

handler = Mangum(app, lifespan="off")
```

### Google Cloud Platform

#### Using Google Cloud Run

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/contract-system', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/contract-system']
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'contract-system'
      - '--image'
      - 'gcr.io/$PROJECT_ID/contract-system'
      - '--platform'
      - 'managed'
      - '--region'
      - 'us-central1'
      - '--allow-unauthenticated'
```

Deploy:
```bash
gcloud builds submit --config cloudbuild.yaml
```

## Configuration Management

### Environment-Specific Configurations

#### Development
```bash
CONTRACT_DEBUG=true
CONTRACT_OPENAI_TEMPERATURE=0.2
CONTRACT_MAX_REGENERATION_ATTEMPTS=2
CONTRACT_QUALITY_GATE_ENABLED=true
CONTRACT_MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

#### Staging
```bash
CONTRACT_DEBUG=false
CONTRACT_OPENAI_TEMPERATURE=0.1
CONTRACT_MAX_REGENERATION_ATTEMPTS=3
CONTRACT_QUALITY_GATE_ENABLED=true
CONTRACT_MLFLOW_TRACKING_URI=postgresql://mlflow-staging:5432/mlflow
```

#### Production
```bash
CONTRACT_DEBUG=false
CONTRACT_OPENAI_TEMPERATURE=0.05
CONTRACT_MAX_REGENERATION_ATTEMPTS=3
CONTRACT_QUALITY_GATE_ENABLED=true
CONTRACT_MLFLOW_TRACKING_URI=postgresql://mlflow-prod:5432/mlflow
```

### Secrets Management

#### Using AWS Secrets Manager
```python
import boto3
import json

def get_secret(secret_name):
    session = boto3.session.Session()
    client = session.client('secretsmanager', region_name='us-east-1')
    
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        return json.loads(get_secret_value_response['SecretString'])
    except Exception as e:
        raise e

# Usage in settings.py
secrets = get_secret('contract-system/prod')
openai_api_key = secrets['openai_api_key']
```

#### Using Kubernetes Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: contract-system-secrets
type: Opaque
data:
  openai-api-key: <base64-encoded-api-key>
```

## Monitoring and Logging

### Application Monitoring

#### Health Checks
```python
# Custom health check endpoint
@app.get("/health/detailed")
async def detailed_health_check():
    checks = {
        "database": check_database_connection(),
        "mlflow": check_mlflow_connection(),
        "openai": check_openai_api(),
        "disk_space": check_disk_space(),
        "memory": check_memory_usage()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
    )
```

#### Prometheus Metrics
```python
# Add to main.py
from prometheus_client import Counter, Histogram, generate_latest

contract_requests = Counter('contract_requests_total', 'Total contract requests')
contract_generation_time = Histogram('contract_generation_seconds', 'Contract generation time')
quality_scores = Histogram('contract_quality_scores', 'Contract quality scores')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Logging Configuration

```python
# logging_config.py
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "formatter": "json",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,
            "backupCount": 5,
        }
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["default", "file"],
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

## Security Considerations

### API Security

1. **Rate Limiting**
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   from slowapi.errors import RateLimitExceeded
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
   
   @app.post("/contracts/generate")
   @limiter.limit("10/minute")
   async def generate_contract(request: Request, ...):
       ...
   ```

2. **Input Validation**
   ```python
   from pydantic import validator
   
   class ContractGenerationRequest(BaseModel):
       contract_data: Dict[str, Any]
       
       @validator('contract_data')
       def validate_contract_data(cls, v):
           # Sanitize input data
           for key, value in v.items():
               if isinstance(value, str):
                   v[key] = html.escape(value)
           return v
   ```

3. **HTTPS Enforcement**
   ```python
   from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
   
   if not settings.debug:
       app.add_middleware(HTTPSRedirectMiddleware)
   ```

### Infrastructure Security

1. **Firewall Configuration**
   ```bash
   # Allow only necessary ports
   sudo ufw default deny incoming
   sudo ufw default allow outgoing
   sudo ufw allow ssh
   sudo ufw allow 80
   sudo ufw allow 443
   sudo ufw enable
   ```

2. **SSL/TLS Configuration**
   ```nginx
   server {
       listen 443 ssl http2;
       ssl_certificate /etc/ssl/certs/your-cert.pem;
       ssl_certificate_key /etc/ssl/private/your-key.pem;
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
       ssl_prefer_server_ciphers off;
       ssl_dhparam /etc/ssl/certs/dhparam.pem;
   }
   ```

### Data Security

1. **Database Encryption**
   ```python
   # Use encrypted connections
   DATABASE_URL = "postgresql://user:pass@host:5432/db?sslmode=require"
   ```

2. **File Storage Security**
   ```python
   # Encrypt sensitive files
   from cryptography.fernet import Fernet
   
   def encrypt_file(file_path: str, key: bytes):
       f = Fernet(key)
       with open(file_path, 'rb') as file:
           file_data = file.read()
       encrypted_data = f.encrypt(file_data)
       with open(file_path + '.encrypted', 'wb') as file:
           file.write(encrypted_data)
   ```

## Backup and Recovery

### Database Backup
```bash
#!/bin/bash
# backup_db.sh
BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
mkdir -p $BACKUP_DIR

pg_dump -h localhost -U postgres contracts > $BACKUP_DIR/contracts.sql
pg_dump -h localhost -U postgres mlflow > $BACKUP_DIR/mlflow.sql

# Compress and upload to S3
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
aws s3 cp $BACKUP_DIR.tar.gz s3://your-backup-bucket/
```

### Application Backup
```bash
#!/bin/bash
# backup_app.sh
rsync -av --exclude='venv' --exclude='__pycache__' \
  /home/contractapp/contract-system/ \
  /backups/app/$(date +%Y-%m-%d)/
```

### Disaster Recovery Plan

1. **Recovery Time Objective (RTO)**: 4 hours
2. **Recovery Point Objective (RPO)**: 1 hour
3. **Backup Schedule**: Daily automated backups
4. **Recovery Procedures**: Documented step-by-step restoration process

## Performance Optimization

### Application Optimization

1. **Connection Pooling**
   ```python
   from sqlalchemy.pool import QueuePool
   
   engine = create_engine(
       DATABASE_URL,
       poolclass=QueuePool,
       pool_size=20,
       max_overflow=30,
       pool_pre_ping=True
   )
   ```

2. **Caching**
   ```python
   import redis
   from functools import wraps
   
   redis_client = redis.Redis(host='localhost', port=6379, db=0)
   
   def cache_result(expire_time=3600):
       def decorator(func):
           @wraps(func)
           async def wrapper(*args, **kwargs):
               cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
               cached_result = redis_client.get(cache_key)
               
               if cached_result:
                   return json.loads(cached_result)
               
               result = await func(*args, **kwargs)
               redis_client.setex(cache_key, expire_time, json.dumps(result))
               return result
           return wrapper
       return decorator
   ```

### Infrastructure Optimization

1. **Load Balancing**
   ```nginx
   upstream contract_backend {
       server 127.0.0.1:8000 weight=3;
       server 127.0.0.1:8001 weight=2;
       server 127.0.0.1:8002 weight=1;
   }
   
   server {
       location / {
           proxy_pass http://contract_backend;
       }
   }
   ```

2. **CDN Configuration**
   ```nginx
   location /static/ {
       expires 1y;
       add_header Cache-Control "public, immutable";
       add_header X-Content-Type-Options "nosniff";
   }
   ```

This deployment guide provides comprehensive instructions for deploying the Lawyer Contract Creation System in various environments while maintaining the quality-focused approach specified in the PRD.