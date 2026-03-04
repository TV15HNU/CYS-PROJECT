import sys
import os

print("Starting local_demo.py verbose version...")
try:
    print("Importing cv2...")
    import cv2
    print("Importing torch...")
    import torch
    print("Importing pyzbar...")
    from pyzbar.pyzbar import decode as zbar_decode
    print("Importing core modules...")
    from core.qr_processing import QRProcessor
    from core.url_analyzer import URLAnalyzer
    from core.ml_models import PhishXCore
    from core.decision_engine import DecisionEngine
    print("All imports successful.")
except ImportError as e:
    print(f"❌ Dependency Error: {e}")
    print("Try running: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected Error during import: {e}")
    sys.exit(1)

def run_local_demo(image_path):
    print(f"\n🚀 PhishX v2 Local Demo: Scanning {image_path}\n")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return

    # 1. QR Processor
    print("Step 1: QR Decoding & Enhancement...")
    qr_processor = QRProcessor()
    qr_results = qr_processor.process_qr(image_path)
    if qr_results["status"] != "success":
        print(f"❌ {qr_results['message']}")
        return
    
    url = qr_results["url"]
    print(f"✅ Extracted URL: {url}")

    # 2. URL Analyzer
    print("Step 2: URL Expansion & Analysis...")
    url_analyzer = URLAnalyzer()
    url_data = url_analyzer.process_url(url)
    final_url = url_data["final_url"]
    print(f"✅ Final Destination: {final_url}")

    # 3. PhishX Core (Predictions)
    print("Step 3: Bayesian Multi-modal Detection (MC Dropout)...")
    print("Note: If first time, this will download 'distilbert-base-uncased' (approx 250MB)...")
    phishx_core = PhishXCore()
    risk, uncertainty, alpha = phishx_core.predict(final_url)
    print(f"✅ Risk Score: {risk:.4f}")
    print(f"✅ Uncertainty: {uncertainty:.4f}")
    print(f"✅ Gating Weight: {alpha:.4f}")

    # 4. Decision Engine
    print("Step 4: Final Verdict...")
    decision_engine = DecisionEngine()
    result = decision_engine.process(risk, uncertainty)
    
    print("\n" + "="*40)
    print(f"RESULT:      {result['prediction'].upper()}")
    print(f"ACTION:      {result['action'].upper()}")
    print(f"CONFIDENCE:  {1 - result['uncertainty']:.2%}")
    print("="*40)

if __name__ == "__main__":
    test_file = r"c:\Users\vishn\OneDrive\Desktop\cyber security\scan_20260211_105544_target.jpg"
    run_local_demo(test_file)
