# Docker Deployment Guide

## Quick Start

```bash
docker compose up -d
docker compose logs -f
curl http://localhost:8000/api/v1/health
docker compose down
```

## Build and Run

```bash
# Using docker-compose
docker compose build
docker compose up -d

# Or manually
docker build -t stereo-vision .
docker run -d -p 8000:8000 stereo-vision
```

## Configuration

Mount volumes:

```yaml
volumes:
  - ./calibration:/app/calibration:ro
  - ./outputs:/app/outputs
```

## Edge Deployment

### NVIDIA Jetson

```bash
docker run -d --runtime nvidia -p 8000:8000 stereo-vision:jetson
```

### Raspberry Pi

```bash
docker buildx build --platform linux/arm64 -t stereo-vision:rpi .
docker run -d -p 8000:8000 --memory=2g stereo-vision:rpi
```

## Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: "4"
      memory: 4G
    reservations:
      cpus: "2"
      memory: 2G
```

## Troubleshooting

**Container wont start:**
```bash
docker compose logs stereo-vision-api
```

**Port in use:**
```bash
docker compose down
lsof -i :8000
```

**Out of memory:**
Increase memory limit in docker-compose.yml to 8G

## Testing

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "left_image=@test_left.png" \
  -F "right_image=@test_right.png"
```

## Monitoring

```bash
# Container stats
docker stats stereo-vision-api

# Health status
docker inspect stereo-vision-api | grep Health
```

## Updates

```bash
git pull
docker compose build
docker compose up -d
```
