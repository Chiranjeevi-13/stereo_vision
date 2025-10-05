# Stereo Vision Perception System

Real-time 3D object detection and localization using stereo vision and deep learning.

**Built in 7 hours** | **Performance: 8-10 FPS on Apple M4**

## What It Does

Takes two camera images → Computes depth → Detects objects → Calculates 3D positions

## Features

- **Stereo depth estimation** using SGBM algorithm
- **Object detection** with YOLOv8 (80 object classes)
- **3D localization** - X, Y, Z coordinates in meters
- **Point cloud generation** - 303K points from scene
- **REST API** - FastAPI with automatic docs
- **Docker deployment** - Runs on edge devices

## Performance

| Component | Time | FPS |
|-----------|------|-----|
| Stereo Depth | 47ms | 21 |
| YOLO Detection | 54ms | 19 |
| 3D Localization | 0.1ms | 10000 |
| **Full Pipeline** | **114ms** | **8.8** |

Accurate to ±20cm at 20 meters.

## Quick Start

```bash
git clone https://github.com/Chiranjeevi-13/stereo_vision.git
cd stereo_vision
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python demo_pipeline.py
```

## Docker

```bash
docker compose up -d
curl http://localhost:8000/api/v1/health
```

## API Example

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "left_image=@left.png" \
  -F "right_image=@right.png"
```

Returns JSON with detected objects and their 3D positions.

## Technology

OpenCV • YOLOv8 • PyTorch • FastAPI • Docker

Tested on KITTI dataset, deployable on Jetson Nano and Raspberry Pi.

## Project Structure

```
backend/        - FastAPI server
perception/     - Vision algorithms (depth, detection, localization)
calibration/    - Camera parameters
pipeline/       - Integrated system
```

## Use Cases

Autonomous robots, self-driving vehicles, drones, warehouse automation.

## Author

**Chiranjeevi Dabbiru**

GitHub: [@Chiranjeevi-13](https://github.com/Chiranjeevi-13)

MIT License
