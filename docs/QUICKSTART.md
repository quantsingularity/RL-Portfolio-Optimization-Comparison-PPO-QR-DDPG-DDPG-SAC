# Quick Start Guide

## Getting Started in 5 Minutes

### Prerequisites

- Docker and Docker Compose installed
- (Optional) NVIDIA GPU with Docker GPU support for training
- 8GB RAM minimum, 16GB recommended

### Step 1: Clone and Setup

```bash
# Clone repository
git clone <your-repo-url>
cd enhanced-rl-portfolio

# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional)
nano .env
```

### Step 2: Start Services

```bash
# Start all services
docker-compose up -d

# Check services are running
docker-compose ps
```

You should see:

- ✅ rl-portfolio-api (FastAPI)
- ✅ rl-portfolio-db (PostgreSQL)
- ✅ rl-portfolio-redis
- ✅ rl-portfolio-grafana
- ✅ rl-portfolio-jupyter

### Step 3: Access Services

Open your browser:

1. **API Documentation**: http://localhost:8000/docs
2. **Jupyter Notebooks**: http://localhost:8888
3. **Grafana Dashboard**: http://localhost:3000 (admin/admin123)

### Step 4: Train Your First Model

```bash
# Option A: Using Docker
docker-compose up training

# Option B: Using local Python
docker exec -it rl-portfolio-api python code/train.py
```

### Step 5: Get Portfolio Recommendations

```bash
# Test API
curl -X POST "http://localhost:8000/api/v1/portfolio/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "test_001",
    "risk_tolerance": "medium",
    "investment_amount": 100000
  }'
```

## Next Steps

1. **Explore Notebooks**: Open Jupyter and check out:
   - `transaction_cost_analysis.ipynb`
   - `reward_ablation.ipynb`
   - `regime_analysis.ipynb`

2. **Customize Configuration**: Edit `config/config.yaml`

3. **Run Experiments**:

   ```bash
   # Transaction cost analysis
   docker exec -it rl-portfolio-api python code/transaction_cost_analysis.py

   # Reward ablation study
   docker exec -it rl-portfolio-api python code/reward_ablation.py

   # Market regime analysis
   docker exec -it rl-portfolio-api python code/regime_analysis.py
   ```

4. **Monitor Performance**: Check Grafana dashboards

## Troubleshooting

### Issue: Services won't start

```bash
# Check logs
docker-compose logs api
docker-compose logs postgres

# Restart services
docker-compose restart
```

### Issue: GPU not detected

```bash
# Check GPU availability
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If not working, install nvidia-docker2
```

### Issue: Port already in use

Edit `docker-compose.yml` and change port mappings:

```yaml
ports:
  - "8001:8000" # Change 8000 to 8001
```

## Configuration Tips

### For Production Use:

1. **Change default passwords** in `.env`
2. **Configure email alerts** (SMTP settings)
3. **Set up SSL/TLS** for API
4. **Configure database backups**
5. **Set resource limits** in docker-compose.yml

### For Development:

1. **Enable hot reload**:

   ```yaml
   command: uvicorn production.api:app --reload
   ```

2. **Mount code directory**:
   ```yaml
   volumes:
     - ./code:/app/code
   ```

## Support

- 📖 Read full documentation in README.md
- 🐛 Report issues on GitHub
- 💬 Ask questions in discussions

## Quick Commands Reference

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f api

# Restart service
docker-compose restart api

# Access shell
docker exec -it rl-portfolio-api bash

# Run tests
docker exec -it rl-portfolio-api pytest

# Backup database
docker exec rl-portfolio-db pg_dump -U portfolio_user portfolio_db > backup.sql

# Restore database
docker exec -i rl-portfolio-db psql -U portfolio_user portfolio_db < backup.sql
```
