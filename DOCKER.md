# Docker Deployment Guide

Complete guide for deploying the stereo vision system using Docker.

## Quick Start
```bash
# Build and start
docker compose up -d

# View logs
docker compose logs -f stereo-vision-api

# Check health
curl http://localhost:8000/api/v1/health

# Stop
docker compose down
