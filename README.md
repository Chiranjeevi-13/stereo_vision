# Stereo Vision Perception System

Real-time 3D object detection and localization using stereo vision and deep learning.

**Development Time:** 7 hours of focused work  
**Performance:** 8-10 FPS on CPU (Apple M4)

## Features

- **Stereo Depth Estimation** - Semi-Global Block Matching (SGBM) algorithm
- **Object Detection** - YOLOv8-nano for real-time detection (80 COCO classes)
- **3D Localization** - Calculates X, Y, Z positions in meters for detected objects
- **Point Cloud Generation** - Dense 3D scene reconstruction with color
- **REST API** - FastAPI backend with automatic documentation
- **Docker Deployment** - Containerized for edge devices and cloud

## Performance Metrics

| Component | Time (ms) | FPS | Accuracy |
|-----------|-----------|-----|----------|
| Stereo Disparity | 47 | 21 | Sub-pixel precision |
| Depth Map | 2 | 500 | 5-50m range |
| Object Detection | 54 | 19 | 37.3 mAP |
| 3D Localization | 0.1 | 10000 | ±20cm @ 20m |
| Point Cloud | 11 | 91 | 303K → 10K points |
| **Full Pipeline** | **114** | **8.8** | - |

## Quick Start

### Prerequisites

- Python 3.11+
- 8GB RAM minimum
- macOS/Linux (tested on macOS)

### Installation
```bash
# Clone repository
git clone https://github.com/Chiranjeevi-13/stereo_vision.git
cd stereo_vision

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

