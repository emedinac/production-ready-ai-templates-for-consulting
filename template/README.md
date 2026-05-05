# AI Project Template

A production-ready template for machine learning projects using DVC, Poetry, and modern MLOps practices.

## Features

- **Data Version Control**: DVC for tracking data, models, and experiments
- **Dependency Management**: Poetry for Python package management
- **Containerization**: Docker Compose for local development and production deployment
- **Experiment Tracking**: MLflow integration for model versioning and metrics
- **Monitoring**: Prometheus and Grafana for system monitoring
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Code Quality**: Pre-configured linting and testing setup

## Quick Start

### Development Setup

```bash
# Install dependencies
make setup

# Run tests
make test

# Run the full pipeline
make pipeline

# Start development server
make serve
```

### Local PostgreSQL Setup (Alternative to S3)

If you prefer to use PostgreSQL for local data storage instead of S3:

```bash
# Start PostgreSQL databases
docker-compose -f docker-compose.prod.yml up -d db

# Initialize DVC with PostgreSQL remotes
make init-prod-postgres

# Or manually configure:
dvc remote add -d storage postgresql://user:password@localhost:5432/dvc_cache
dvc remote add models postgresql://user:password@localhost:5432/models
dvc remote add data postgresql://user:password@localhost:5432/data

# Then run the pipeline
make pipeline-prod
```

### Production Setup

1. **Configure Environment Variables**
   ```bash
   cp .env.prod.example .env.prod
   # Edit .env.prod with your production values
   ```

2. **Choose Your Storage Backend**

   **Option A: S3 (Cloud Storage)**
   ```bash
   make init-prod-s3

   # Or manually configure:
   dvc remote add -d storage s3://your-production-bucket/dvc-cache
   dvc remote add models s3://your-models-bucket/models
   dvc remote add data s3://your-data-bucket/data
   ```

   **Option B: PostgreSQL (Local Database)**
   ```bash
   # Start PostgreSQL database
   docker-compose -f docker-compose.prod.yml up -d db

   # Initialize DVC with PostgreSQL remotes
   make init-prod-postgres

   # Or manually configure:
   dvc remote add -d storage postgresql://user:password@localhost:5432/dvc_cache
   dvc remote add models postgresql://user:password@localhost:5432/models
   dvc remote add data postgresql://user:password@localhost:5432/data
   ```

3. **Deploy to Production**
   ```bash
   # Build and deploy
   docker-compose -f docker-compose.prod.yml up -d

   # Or use the Makefile target
   make deploy-model
   ```

## Project Structure

```
{{project_name}}/
├── configs/           # Configuration files
├── data/             # Data directory (tracked by DVC)
│   ├── raw/         # Raw input data
│   └── processed/   # Processed data
├── evaluation/      # Model evaluation scripts
├── experiments/     # Experiment configurations
├── models/          # Model artifacts and checkpoints
├── monitoring/      # Monitoring and alerting
├── pipelines/       # DVC pipeline helpers
├── serving/         # Model serving API
├── src/            # Source code
├── tests/          # Unit and integration tests
└── utils/          # Utility functions
```

## DVC Pipeline

The project uses DVC to manage the ML pipeline with the following stages:

1. **preprocess**: Data preprocessing and feature engineering
2. **train**: Model training with hyperparameter optimization
3. **evaluate**: Model evaluation and performance metrics

### Running the Pipeline

```bash
# Development
make pipeline

# Production (with remote data)
make pipeline-prod

# Run specific stage
poetry run python -m {{project_name}}.pipelines.dvc_pipeline run_stage preprocess
```

### Data Management

```bash
# Pull latest data from remote
make data-pull

# Push processed data to remote
make data-push

# Push trained models
make model-push
```

## Configuration

### Parameters

Edit `params.yaml` to configure:
- Experiment settings
- Data source configuration
- Model hyperparameters
- Evaluation metrics

### DVC Configuration

The `.dvc/config` file contains remote storage settings. Choose one of the following configurations:

**S3 Configuration (Cloud):**
```ini
['remote "storage"']
    url = s3://your-production-bucket/dvc-cache
    access_key_id = ${AWS_ACCESS_KEY_ID}
    secret_access_key = ${AWS_SECRET_ACCESS_KEY}
    region = us-east-1
['remote "models"']
    url = s3://your-models-bucket/models
    access_key_id = ${AWS_ACCESS_KEY_ID}
    secret_access_key = ${AWS_SECRET_ACCESS_KEY}
    region = us-east-1
['remote "data"']
    url = s3://your-data-bucket/data
    access_key_id = ${AWS_ACCESS_KEY_ID}
    secret_access_key = ${AWS_SECRET_ACCESS_KEY}
    region = us-east-1
[core]
    remote = storage
```

**PostgreSQL Configuration (Local):**
```ini
['remote "storage"']
    url = postgresql://user:password@localhost:5432/dvc_cache
['remote "models"']
    url = postgresql://user:password@localhost:5432/models
['remote "data"']
    url = postgresql://user:password@localhost:5432/data
[core]
    remote = storage
```

## Deployment

### Local Development

```bash
docker-compose up -d
```

### Production Deployment

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### API Endpoints

Once deployed, the API will be available at:
- **API**: http://localhost:8000
- **MLflow UI**: http://localhost:5000
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

## Monitoring

### Data Drift Detection

```bash
make monitor
```

### Metrics

- Model performance metrics are tracked in MLflow
- System metrics are collected by Prometheus
- Dashboards available in Grafana

## CI/CD

The project includes GitHub Actions workflows for:
- Automated testing on pull requests
- DVC pipeline execution on main branch
- Model deployment to production

### Required Secrets

Set these in your GitHub repository secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `MLFLOW_TRACKING_URI`

## Development Guidelines

### Code Quality

```bash
# Run tests
make test

# Validate configuration
make validate-config

# Check pipeline
poetry run python -m {{project_name}}.pipelines.dvc_pipeline validate_pipeline
```

### Adding New Stages

1. Add the stage to `dvc.yaml`
2. Implement the stage logic in `src/`
3. Update the pipeline helpers if needed
4. Test the new stage

### Model Promotion

```bash
# Promote model to production
make promote
```

## Troubleshooting

### Common Issues

1. **DVC remote connection failed (S3)**
   - Check AWS credentials in `.env.prod`
   - Verify S3 bucket permissions
   - Ensure AWS region is correct

2. **DVC remote connection failed (PostgreSQL)**
   - Verify PostgreSQL is running: `docker-compose -f docker-compose.prod.yml ps db`
   - Check database connection: `psql postgresql://user:password@localhost:5432/dvc_cache -c "SELECT 1"`
   - Ensure databases are created: check `scripts/init-dvc-databases.sql`

3. **Pipeline fails**
   - Check `dvc status` for stage status
   - Review logs in `dvc status --show-json`
   - Verify dependencies exist in configured remote

4. **Model deployment fails**
   - Verify model artifacts exist in remote storage
   - Check API health endpoint
   - Ensure correct MODEL_VERSION in environment

5. **PostgreSQL connection issues**
   - Check database logs: `docker-compose -f docker-compose.prod.yml logs db`
   - Verify connection string format
   - Ensure PostgreSQL port (5432) is accessible

### Logs

- Application logs: Check Docker container logs
- DVC logs: `dvc status`
- MLflow logs: Available in MLflow UI

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and validation
5. Submit a pull request

## License

This project is licensed under the MIT License.