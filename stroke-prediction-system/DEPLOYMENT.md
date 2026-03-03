# 🚀 Deployment Guide

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Checklist](#production-checklist)
5. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Local Development

### Quick Start
```bash
# 1. Clone and setup
git clone <repo-url>
cd stroke-prediction-system
chmod +x setup.sh
./setup.sh

# 2. Start services
# Terminal 1: API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Dashboard
streamlit run dashboard/app.py
```

### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models
cd ml && python train.py && cd ..

# Run tests
pytest tests/ -v
```

---

## Docker Deployment

### Development
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production
```bash
# Use production docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Health check
curl http://localhost:8000/health
```

---

## Cloud Deployment

### AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p docker stroke-prediction-api --region us-east-1

# Create environment
eb create stroke-prediction-prod

# Deploy updates
eb deploy

# Open application
eb open

# View logs
eb logs

# SSH into instance
eb ssh
```

### AWS ECS (Fargate)

```bash
# Build and push to ECR
aws ecr create-repository --repository-name stroke-prediction-api

# Tag and push image
docker build -t stroke-prediction-api .
docker tag stroke-prediction-api:latest <ecr-url>/stroke-prediction-api:latest
docker push <ecr-url>/stroke-prediction-api:latest

# Create ECS task definition and service (use AWS Console or CloudFormation)
```

### Google Cloud Platform

```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT-ID/stroke-prediction-api

# Deploy
gcloud run deploy stroke-prediction-api \
  --image gcr.io/PROJECT-ID/stroke-prediction-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Get URL
gcloud run services describe stroke-prediction-api --region us-central1
```

### Heroku

```bash
# Login
heroku login

# Create app
heroku create stroke-prediction-api

# Deploy with Docker
heroku container:push web -a stroke-prediction-api
heroku container:release web -a stroke-prediction-api

# Open app
heroku open -a stroke-prediction-api

# View logs
heroku logs --tail -a stroke-prediction-api
```

### Azure

```bash
# Login
az login

# Create resource group
az group create --name stroke-prediction-rg --location eastus

# Create container registry
az acr create --resource-group stroke-prediction-rg \
  --name strokepredictionacr --sku Basic

# Build and push
az acr build --registry strokepredictionacr \
  --image stroke-prediction-api:latest .

# Deploy to Container Instances
az container create \
  --resource-group stroke-prediction-rg \
  --name stroke-prediction-api \
  --image strokepredictionacr.azurecr.io/stroke-prediction-api:latest \
  --dns-name-label stroke-prediction-api \
  --ports 8000
```

---

## Production Checklist

### Security
- [ ] Enable HTTPS (use Nginx reverse proxy or cloud load balancer)
- [ ] Add authentication (JWT tokens, OAuth)
- [ ] Set up CORS with specific allowed origins
- [ ] Implement rate limiting
- [ ] Add API keys for external access
- [ ] Set up secrets management (AWS Secrets Manager, etc.)
- [ ] Regular security updates

### Performance
- [ ] Set up caching (Redis)
- [ ] Configure auto-scaling
- [ ] Optimize model loading (lazy loading)
- [ ] Add database for prediction logging
- [ ] Implement connection pooling
- [ ] Configure CDN for static assets

### Reliability
- [ ] Set up health checks
- [ ] Configure logging (centralized)
- [ ] Add error tracking (Sentry)
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure alerting
- [ ] Implement circuit breakers
- [ ] Set up backup and recovery

### Observability
- [ ] Application metrics (response time, error rate)
- [ ] Model metrics (prediction distribution, confidence)
- [ ] Infrastructure metrics (CPU, memory, disk)
- [ ] Custom business metrics
- [ ] Distributed tracing (Jaeger, Zipkin)

### CI/CD
```yaml
# Example GitHub Actions workflow
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Your deployment commands
```

---

## Monitoring & Maintenance

### Health Monitoring

```bash
# API health check
curl http://your-domain.com/health

# Expected response
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Log Monitoring

```bash
# View API logs
docker-compose logs -f api

# View specific timeframe
docker-compose logs --since 10m api
```

### Model Monitoring

**Key Metrics to Track:**
1. Prediction distribution (% positive predictions)
2. Average confidence score
3. Response time (latency)
4. Error rate
5. Feature drift detection

**Set Alerts For:**
- Prediction distribution shifts >10%
- Average confidence drops <70%
- Response time >500ms
- Error rate >1%
- Model failures

### Updating the Model

```bash
# 1. Train new model
cd ml
python train.py

# 2. Test new model
pytest tests/

# 3. Backup old model
cp models/stroke_model_production.pkl models/stroke_model_production_backup.pkl

# 4. Deploy new model
# Copy new model to production
# Restart services
docker-compose restart api

# 5. Monitor for issues
# Watch metrics for 24-48 hours
```

### Rollback Procedure

```bash
# If new model has issues:
# 1. Stop services
docker-compose down

# 2. Restore backup
cp models/stroke_model_production_backup.pkl models/stroke_model_production.pkl

# 3. Restart
docker-compose up -d

# 4. Verify
curl http://localhost:8000/health
```

---

## Environment Variables

### Required
```bash
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/stroke_model_production.pkl
```

### Optional
```bash
LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:8501"]
RATE_LIMIT=100  # requests per minute
CACHE_TTL=300   # seconds
```

### Example .env file
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Model Configuration
MODEL_PATH=models/stroke_model_production.pkl
PREPROCESSOR_PATH=models/preprocessor.pkl

# Security
API_KEY_REQUIRED=true
JWT_SECRET=your-secret-key-here

# Performance
ENABLE_CACHE=true
CACHE_TTL=300
WORKERS=4

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_ENABLED=true
```

---

## Troubleshooting

### Common Issues

**Issue: Model not loading**
```bash
# Check if models exist
ls -la models/

# Retrain if needed
cd ml && python train.py
```

**Issue: API not responding**
```bash
# Check if port is in use
lsof -i :8000

# Check container logs
docker-compose logs api
```

**Issue: Out of memory**
```bash
# Increase Docker memory limit
# In docker-compose.yml:
services:
  api:
    mem_limit: 2g
```

**Issue: Slow predictions**
```bash
# Enable model caching
# Add Redis caching layer
# Optimize model loading (lazy loading)
```

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs`
2. Review API docs: `http://localhost:8000/docs`
3. Run health check: `curl http://localhost:8000/health`
4. Check GitHub Issues
5. Contact: your.email@example.com

---

**Last Updated:** February 2025
