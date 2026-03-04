# PhishX v2: Robust QR Phishing Detection System

## 📌 Project Overview
PhishX v2 is an end-to-end phishing detection pipeline designed specifically for QR codes. It handles stylized, occluded, or damaged QR codes using robust image enhancement and a multi-decoder strategy. Extracted URLs are then analyzed using a multi-modal Bayesian ML model that estimates both phishing risk and predictive uncertainty.

## 🔷 System Architecture
1. **QR Acquisition Service**: FastAPI endpoint accepting QR images.
2. **Image Recovery & Enhancement**: OpenCV-based preprocessing (adaptive thresholding, CLAHE, morphological closing).
3. **Multi-Decoder QR Extraction**: Use of ZBar, OpenCV, and WeChat QR decoders for high-success extraction.
4. **URL Validation & Expansion**: Following redirects and expanding shortenings (bit.ly, tinyurl).
5. **Feature Extraction Engine**: Generating 10 structural features from the final URL.
6. **PhishX ML Core**:
   - **Semantic Engine (DistilBERT)**: Deep semantic analysis of the URL string.
   - **Structural Engine (Character CNN)**: Analysis of character-level patterns.
   - **Bayesian Uncertainty**: MC Dropout (10 passes) to compute risk mean and variance.
7. **Adaptive Gating Network**: Dynamic weighting of predictions based on model confidence.
8. **Risk Fusion & Decision Engine**: Final verdict (Safe, Warning, Block) based on risk and uncertainty thresholds.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (with CUDA support for faster inference)
- libzbar (system library)

### Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   python main.py
   ```
   Or use Uvicorn:
   ```bash
   uvicorn main:app --reload
   ```

### API Usage
- **Endpoint**: `POST /scan`
- **Payload**: Form-data with `file` (image)
- **Response**:
  ```json
  {
    "status": "success",
    "url": "original_qr_url",
    "final_url": "expanded_url",
    "prediction": "phishing",
    "risk_score": 0.87,
    "uncertainty": 0.03,
    "action": "block"
  }
  ```

## 🏗 Project Structure
- `api/`: FastAPI routes and request/response models.
- `core/`: Logic for QR processing, URL analysis, and ML models.
- `models/`: Pre-trained weights for Transformer, CNN, and Gating Network.
- `uploads/`: Temporary storage for scanned images.
- `utils/`: Helper functions for logging and metrics.

## 🛡 Security Features
- **Anti-Adversarial**: Bayesian uncertainty detects out-of-distribution (adversarial) URLs.
- **Robust Extraction**: Handles stylized QRs that standard decoders often fail on.
- **Redirection Safety**: Analyzes the final destination URL, not just the intermediate short link.
