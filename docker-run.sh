#!/bin/bash

# Nekkanti OCR Docker Deployment Script
echo "🏦 Nekkanti OCR - Docker Deployment"
echo "=================================="

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker and try again."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to clean up previous containers
cleanup() {
    echo "🧹 Cleaning up previous containers..."
    docker-compose down --remove-orphans
    docker system prune -f
}

# Function to build and start services
start_services() {
    echo "🚀 Building and starting services..."
    
    # Load environment variables if .env file exists
    if [ -f "docker.env" ]; then
        echo "📁 Loading environment variables from docker.env"
        export $(cat docker.env | grep -v '^#' | xargs)
    fi
    
    # Build and start containers
    docker-compose up --build -d
    
    echo "⏳ Waiting for services to be healthy..."
    
    # Wait for API to be healthy
    echo "🔍 Checking API health..."
    timeout=120
    counter=0
    while [ $counter -lt $timeout ]; do
        if curl -f http://localhost:8888/health > /dev/null 2>&1; then
            echo "✅ API is healthy"
            break
        fi
        sleep 2
        counter=$((counter + 2))
        echo -n "."
    done
    
    if [ $counter -ge $timeout ]; then
        echo "❌ API failed to start within ${timeout} seconds"
        docker-compose logs api
        exit 1
    fi
    
    # Wait for Streamlit to be healthy
    echo "🔍 Checking Streamlit health..."
    counter=0
    while [ $counter -lt $timeout ]; do
        if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
            echo "✅ Streamlit is healthy"
            break
        fi
        sleep 2
        counter=$((counter + 2))
        echo -n "."
    done
    
    if [ $counter -ge $timeout ]; then
        echo "❌ Streamlit failed to start within ${timeout} seconds"
        docker-compose logs streamlit
        exit 1
    fi
}

# Function to show service status
show_status() {
    echo ""
    echo "🎉 Nekkanti OCR is running!"
    echo "=========================="
    echo "📊 API Server: http://localhost:8888"
    echo "📖 API Documentation: http://localhost:8888/docs"
    echo "🎨 Streamlit App: http://localhost:8501"
    echo ""
    echo "🔐 Default Login Credentials:"
    echo "   - admin / admin123"
    echo "   - user / user123"
    echo "   - demo / demo123"
    echo ""
    echo "📝 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
    echo "🔄 To restart: docker-compose restart"
}

# Main execution
case "${1:-start}" in
    "start")
        check_docker
        cleanup
        start_services
        show_status
        ;;
    "stop")
        echo "🛑 Stopping Nekkanti OCR services..."
        docker-compose down
        echo "✅ Services stopped"
        ;;
    "restart")
        echo "🔄 Restarting Nekkanti OCR services..."
        docker-compose restart
        show_status
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "clean")
        echo "🧹 Cleaning up Docker resources..."
        docker-compose down --volumes --remove-orphans
        docker system prune -a -f
        echo "✅ Cleanup complete"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|clean}"
        echo ""
        echo "Commands:"
        echo "  start   - Build and start all services (default)"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - Show live logs"
        echo "  clean   - Clean up all Docker resources"
        exit 1
        ;;
esac 