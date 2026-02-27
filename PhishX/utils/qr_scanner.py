import cv2
from pyzbar import pyzbar
import base64
import numpy as np

def scan_qr_from_image(base64_img):
    try:
        if ',' in base64_img:
            base64_img = base64_img.split(',')[1]
        
        img_bytes = base64.b64decode(base64_img)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        qr_codes = pyzbar.decode(img)
        if not qr_codes:
            return None
        
        return qr_codes[0].data.decode('utf-8')
    except Exception as e:
        print(f"QR Scanning Error: {e}")
        return None
