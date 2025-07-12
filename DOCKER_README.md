# 🐳 Nekkanti OCR - Docker Setup

This guide explains how to run the Nekkanti OCR system using Docker containers.

## 🏗️ Architecture

The system consists of two main services:

- **Backend API** (`api`): FastAPI server with PaddleOCR for document processing
- **Frontend UI** (`streamlit`): Streamlit web interface with authentication

Both services run in separate containers and communicate through a Docker network.

## 📋 Prerequisites

- Docker (version 20.0 or higher)
- Docker Compose (version 1.28 or higher)
- At least 4GB of RAM available for Docker
- Ports 8888 and 8501 available on your system

## 🚀 Quick Start

### 1. Clone and Navigate to Project
```bash
git clone <repository-url>
cd nekkanti-ocr
```

### 2. Start the System
```bash
# Using the deployment script (recommended)
./docker-run.sh start

# Or using docker-compose directly
docker-compose up --build -d
```

### 3. Access the Applications
- **Streamlit UI**: http://localhost:8501
- **API Server**: http://localhost:8888
- **API Documentation**: http://localhost:8888/docs

## 🔐 Authentication

The Streamlit UI includes static authentication with these default credentials:

| Username | Password | Role |
|----------|----------|------|
| `admin`  | `admin123` | Administrator |
| `user`   | `user123`  | Standard User |
| `demo`   | `demo123`  | Demo User |

### Changing Passwords

You can customize passwords using environment variables:

```bash
# Create a .env file
cat > .env << EOF
ADMIN_PASSWORD=your_secure_admin_password
USER_PASSWORD=your_secure_user_password
DEMO_PASSWORD=your_secure_demo_password
EOF

# Restart the services
./docker-run.sh restart
```

## 🛠️ Management Commands

The `docker-run.sh` script provides convenient management commands:

```bash
# Start all services (build if needed)
./docker-run.sh start

# Stop all services
./docker-run.sh stop

# Restart services
./docker-run.sh restart

# View live logs
./docker-run.sh logs

# Clean up all Docker resources
./docker-run.sh clean
```

## 📊 Manual Docker Compose Usage

If you prefer using docker-compose directly:

```bash
# Build and start services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild a specific service
docker-compose build api
docker-compose up -d api

# Scale services (if needed)
docker-compose up -d --scale api=2
```

## 🔍 Health Checks

Both services include health checks:

- **API Health**: `http://localhost:8888/health`
- **Streamlit Health**: `http://localhost:8501/_stcore/health`

You can check the health status:

```bash
# Check API health
curl http://localhost:8888/health

# Check Streamlit health
curl http://localhost:8501/_stcore/health

# View health status in Docker
docker-compose ps
```

## 📁 Data Persistence

The system uses Docker volumes to persist data:

```yaml
volumes:
  - ./comprehensive_output:/app/comprehensive_output
  - ./bank_statement_output:/app/bank_statement_output
  - ./bs_reconstructed_pdfs:/app/bs_reconstructed_pdfs
  - ./ocr_outputs_reconstructed:/app/ocr_outputs_reconstructed
```

Processed files and outputs are saved to these directories on your host machine.

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_PASSWORD` | `admin123` | Admin user password |
| `USER_PASSWORD` | `user123` | Standard user password |
| `DEMO_PASSWORD` | `demo123` | Demo user password |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8888` | API server port |

### Custom Configuration

Create a `docker.env` file to override defaults:

```bash
# Authentication
ADMIN_PASSWORD=secure_admin_pass
USER_PASSWORD=secure_user_pass
DEMO_PASSWORD=secure_demo_pass

# API Configuration
API_HOST=0.0.0.0
API_PORT=8888
```

## 🐛 Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check Docker is running
docker info

# Check port availability
netstat -tulpn | grep :8888
netstat -tulpn | grep :8501

# Check logs for errors
docker-compose logs api
docker-compose logs streamlit
```

#### Memory Issues
```bash
# Increase Docker memory limit to at least 4GB
# Check current memory usage
docker stats

# Clean up unused resources
docker system prune -a
```

#### Build Failures
```bash
# Clear build cache
docker builder prune -a

# Rebuild from scratch
docker-compose build --no-cache
```

#### PaddleOCR Issues
```bash
# Check if PaddleOCR models are downloading
docker-compose logs api | grep -i paddle

# Ensure sufficient disk space (models are ~100MB)
df -h
```

### Logs and Debugging

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs streamlit

# Follow logs in real-time
docker-compose logs -f

# Check container status
docker-compose ps

# Inspect container details
docker inspect nekkanti-ocr_api_1
```

### Performance Tuning

#### Resource Limits
Add resource limits to `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          memory: 2G
```

#### Volume Optimization
For better performance, use named volumes for temporary data:

```yaml
volumes:
  temp_data:
    driver: local
```

## 🔒 Security Considerations

### Production Deployment

1. **Change Default Passwords**: Always change default passwords in production
2. **Use HTTPS**: Configure reverse proxy with SSL certificates
3. **Network Security**: Limit network exposure using Docker networks
4. **Resource Limits**: Set appropriate CPU and memory limits
5. **Log Management**: Configure log rotation and monitoring

### Network Security
```yaml
networks:
  nekkanti-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 📈 Monitoring

### Basic Monitoring
```bash
# Monitor resource usage
docker stats

# Monitor container health
docker-compose ps

# Check service availability
curl -f http://localhost:8888/health
curl -f http://localhost:8501/_stcore/health
```

### Advanced Monitoring
Consider integrating with:
- Prometheus + Grafana for metrics
- ELK Stack for log aggregation
- Docker Health Check endpoints

## 🚀 Production Deployment

For production deployment:

1. **Use Environment Secrets**: Store passwords in Docker secrets or external secret management
2. **Reverse Proxy**: Use Nginx or Traefik as reverse proxy
3. **SSL/TLS**: Configure HTTPS certificates
4. **Backup Strategy**: Implement regular backups of persistent volumes
5. **Monitoring**: Set up comprehensive monitoring and alerting

Example production docker-compose with Nginx:

```yaml
version: '3.8'
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - api
      - streamlit
  
  # ... rest of services
```

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs using `docker-compose logs`
3. Ensure system requirements are met
4. Verify network connectivity and port availability

For additional support, please refer to the main project documentation or create an issue in the project repository. 