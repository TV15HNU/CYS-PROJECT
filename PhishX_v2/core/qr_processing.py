import cv2
import numpy as np
from pyzbar.pyzbar import decode as zbar_decode

class QRProcessor:
    def __init__(self):
        # Initialize OpenCV QR detector
        self.opencv_detector = cv2.QRCodeDetector()
        try:
            # Attempt to use WeChat QR detector if available (it's more robust)
            # This requires opencv-contrib-python
            self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
            self.has_wechat = True
        except (AttributeError, Exception):
            self.has_wechat = False

    def enhance_image(self, image):
        """
        Robust QR image enhancement pipeline.
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian blur for denoising
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # 4. Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 5. Morphological closing to fill gaps in modules
        kernel = np.ones((3,3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return closed

    def decode(self, image):
        """
        Multi-decoder strategy to maximize success rate.
        """
        # Strategy 1: Original Image with WeChat
        if self.has_wechat:
            res, _ = self.wechat_detector.detectAndDecode(image)
            if res and res[0]:
                return res[0]

        # Strategy 2: ZBar on original
        zbar_res = zbar_decode(image)
        if zbar_res:
            return zbar_res[0].data.decode('utf-8')

        # Strategy 3: OpenCV on original
        res, _, _ = self.opencv_detector.detectAndDecode(image)
        if res:
            return res

        # Strategy 4: Enhanced Image with decoders
        enhanced = self.enhance_image(image)
        
        # Try ZBar on enhanced
        zbar_res = zbar_decode(enhanced)
        if zbar_res:
            return zbar_res[0].data.decode('utf-8')
            
        # Try OpenCV on enhanced
        res, _, _ = self.opencv_detector.detectAndDecode(enhanced)
        if res:
            return res

        return None

    def process_qr(self, image_path):
        """
        High-level wrapper for QR processing.
        """
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "message": "Failed to load image"}
        
        url = self.decode(image)
        if url:
            return {"status": "success", "url": url}
        else:
            return {"status": "decode_failed", "message": "Please rescan or use better lighting"}
