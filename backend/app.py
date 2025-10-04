"""
FastAPI Backend for Stereo Vision Perception System
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import time
from typing import List, Dict
import logging
from io import BytesIO

# Import our existing modules
from pipeline.main_pipeline import StereoVisionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stereo Vision Perception API",
    description="Real-time 3D object detection and localization from stereo images",
    version="1.0.0"
)

# Global pipeline instance (loaded once at startup)
pipeline = None

# Performance metrics storage
metrics = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'avg_processing_time': 0.0,
    'component_times': {
        'disparity': [],
        'depth': [],
        'detection': [],
        'localization': []
    }
}


@app.on_event("startup")
async def startup_event():
    """
    Initialize pipeline on server startup.
    Loads YOLO model and calibration once.
    """
    global pipeline
    
    logger.info("Starting Stereo Vision API...")
    logger.info("Initializing perception pipeline...")
    
    try:
        pipeline = StereoVisionPipeline(
            config_path='calibration/kitti_stereo_params.yaml',
            yolo_model='yolov8n.pt',
            yolo_confidence=0.5
        )
        logger.info("Pipeline initialized successfully")
        logger.info("API ready to accept requests")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Stereo Vision Perception API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "detect": "/api/v1/detect",
            "health": "/api/v1/health",
            "metrics": "/api/v1/metrics",
            "docs": "/docs"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns system status, model availability, and readiness.
    """
    health_status = {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "yolo_model": "loaded" if pipeline and pipeline.detector else "not loaded",
        "calibration": "loaded" if pipeline and pipeline.stereo_params else "not loaded",
        "stereo_matcher": "ready" if pipeline and pipeline.stereo else "not ready",
        "total_requests_processed": metrics['total_requests'],
        "uptime": "running"
    }
    
    # Determine overall health
    if not pipeline:
        health_status['status'] = "unhealthy"
        health_status['error'] = "Pipeline not initialized"
        return JSONResponse(status_code=503, content=health_status)
    
    return health_status


@app.get("/api/v1/metrics")
async def get_metrics():
    """
    Get performance metrics.
    
    Returns per-component timing breakdowns and overall statistics.
    """
    # Calculate average times for each component
    avg_times = {}
    for component, times in metrics['component_times'].items():
        if times:
            avg_times[component] = {
                'avg_ms': np.mean(times) * 1000,
                'min_ms': np.min(times) * 1000,
                'max_ms': np.max(times) * 1000,
                'avg_fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
            }
        else:
            avg_times[component] = {'avg_ms': 0, 'min_ms': 0, 'max_ms': 0, 'avg_fps': 0}
    
    return {
        "total_requests": metrics['total_requests'],
        "successful_requests": metrics['successful_requests'],
        "failed_requests": metrics['failed_requests'],
        "success_rate": metrics['successful_requests'] / metrics['total_requests'] * 100 
                       if metrics['total_requests'] > 0 else 0,
        "avg_processing_time_ms": metrics['avg_processing_time'] * 1000,
        "avg_fps": 1.0 / metrics['avg_processing_time'] if metrics['avg_processing_time'] > 0 else 0,
        "component_performance": avg_times
    }


@app.post("/api/v1/detect")
async def detect_objects(
    left_image: UploadFile = File(..., description="Left camera image"),
    right_image: UploadFile = File(..., description="Right camera image")
):
    """
    Detect and localize objects in 3D from stereo image pair.
    
    Args:
        left_image: Left camera image file
        right_image: Right camera image file
        
    Returns:
        JSON with detected objects, 3D positions, and performance metrics
    """
    metrics['total_requests'] += 1
    start_time = time.time()
    
    try:
        # Validate pipeline is loaded
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Read and decode images
        logger.info(f"Processing request: {left_image.filename}, {right_image.filename}")
        
        left_bytes = await left_image.read()
        right_bytes = await right_image.read()
        
        # Convert bytes to numpy arrays
        left_arr = np.frombuffer(left_bytes, np.uint8)
        right_arr = np.frombuffer(right_bytes, np.uint8)
        
        # Decode images
        left_img = cv2.imdecode(left_arr, cv2.IMREAD_COLOR)
        right_img = cv2.imdecode(right_arr, cv2.IMREAD_COLOR)
        
        # Validate images loaded correctly
        if left_img is None:
            raise HTTPException(status_code=400, detail="Failed to decode left image")
        if right_img is None:
            raise HTTPException(status_code=400, detail="Failed to decode right image")
        
        # Validate image shapes match
        if left_img.shape != right_img.shape:
            raise HTTPException(
                status_code=400, 
                detail=f"Image shape mismatch: left {left_img.shape} vs right {right_img.shape}"
            )
        
        logger.info(f"Images loaded: {left_img.shape}")
        
        # Process through pipeline
        results = pipeline.process_stereo_pair(
            left_img, 
            right_img,
            generate_pc=False,  # Skip point clouds for API (too slow/large)
            save_outputs=False
        )
        
        # Update metrics
        for component, value in results['timings'].items():
            if component in metrics['component_times']:
                metrics['component_times'][component].append(value)
        
        # Format response
        response = {
            "success": True,
            "processing_time_ms": results['timings']['total'] * 1000,
            "fps": results['fps'],
            "image_shape": list(left_img.shape),
            "depth_statistics": {
                "valid_pixels_percent": results['depth_stats']['valid_percentage'],
                "min_depth_m": results['depth_stats']['min_depth'],
                "max_depth_m": results['depth_stats']['max_depth'],
                "mean_depth_m": results['depth_stats']['mean_depth']
            },
            "detected_objects": [
                {
                    "class_name": obj['class_name'],
                    "confidence": obj['confidence'],
                    "bbox_2d": obj['bbox'],
                    "position_3d": {
                        "x": obj['position_3d']['X'],
                        "y": obj['position_3d']['Y'],
                        "z": obj['position_3d']['Z']
                    },
                    "distance_m": obj['distance'],
                    "depth_m": obj['depth']
                }
                for obj in results['objects_3d']
            ],
            "num_objects": len(results['objects_3d']),
            "component_timings_ms": {
                key: value * 1000 for key, value in results['timings'].items()
            }
        }
        
        # Update success metrics
        metrics['successful_requests'] += 1
        total_time = time.time() - start_time
        metrics['avg_processing_time'] = (
            (metrics['avg_processing_time'] * (metrics['successful_requests'] - 1) + total_time) 
            / metrics['successful_requests']
        )
        
        logger.info(f"Request processed successfully: {len(results['objects_3d'])} objects detected in {total_time*1000:.1f}ms")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        metrics['failed_requests'] += 1
        raise
        
    except Exception as e:
        # Log and return internal server error
        metrics['failed_requests'] += 1
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
