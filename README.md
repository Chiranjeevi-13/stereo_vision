# Stereo Vision Perception System

Real-time 3D object detection and localization using stereo vision and deep learning.

**Development Time:** 7 hours | **Performance:** 8-10 FPS on CPU (Apple M4)

## Features

- Stereo depth estimation (SGBM algorithm)
- YOLOv8-nano object detection (80 classes)
- 3D localization with X,Y,Z positions
- Point cloud generation
- FastAPI REST API
- Docker deployment

## Quick Start
```bash
git clone https://github.com/Chiranjeevi-13/stereo_vision.git
cd stereo_vision
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python demo_pipeline.py