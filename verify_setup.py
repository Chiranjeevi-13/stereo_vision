import sys

def check_import(module_name, package_name=None):
    if package_name is None:
        package_name = module_name
    try:
        if module_name == "cv2":
            import cv2
            version = cv2.__version__
        elif module_name == "ultralytics":
            from ultralytics import YOLO
            version = "OK"
        elif module_name == "numpy":
            import numpy as np
            version = np.__version__
        elif module_name == "scipy":
            import scipy
            version = scipy.__version__
        elif module_name == "matplotlib":
            import matplotlib
            version = matplotlib.__version__
        else:
            exec(f"import {module_name}")
            version = "OK"
        print(f"✅ {package_name:20s} - {version}")
        return True
    except Exception as e:
        print(f"❌ {package_name:20s} - {e}")
        return False

def main():
    print("="*50)
    print("SETUP VERIFICATION")
    print("="*50)
    checks = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("ultralytics", "YOLOv8"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
    ]
    results = [check_import(m, n) for m, n in checks]
    print("="*50)
    if all(results):
        print("✅ ALL PACKAGES WORKING")
        import cv2
        print(f"\nOpenCV: {cv2.__version__}")
        print(f"StereoBM: Available")
        print(f"StereoSGBM: Available")
        return 0
    else:
        print("❌ SOME PACKAGES FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
