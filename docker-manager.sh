#!/bin/bash
# MLOps Lab - Docker Manager

case "${1:-help}" in
    start)
        echo "🚀 Starting services..."
        docker-compose up --build -d
        echo "✅ API: http://localhost:8000/docs"
        echo "✅ UI:  http://localhost:8501"
        ;;
    stop)
        echo "🛑 Stopping services..."
        docker-compose down
        ;;
    logs)
        docker-compose logs -f
        ;;
    *)
        echo "Usage: ./docker-manager.sh [start|stop|logs]"
        ;;
esac