import sys
print("Checking imports...")
try:
    import fastapi
    print("FastAPI: OK")
    import uvicorn
    print("Uvicorn: OK")
    import cv2
    print("OpenCV: OK")
    import torch
    print("Torch: OK")
    from pyzbar.pyzbar import decode
    print("PyZBar: OK")
    import transformers
    print("Transformers: OK")
except Exception as e:
    print(f"FAILED IMPORT: {e}")
    sys.exit(1)
print("All vital imports are OK!")
