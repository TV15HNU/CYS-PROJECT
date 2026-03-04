import os
import shutil
import uuid
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import original PhishX logic from the wrapper
from core.original_wrapper import (
    URLTransformer, CharCNN, CharTokenizer, 
    UncertaintyAwareGating, PhishXEnsemble,
    get_transformer_tokenizer, extract_numeric_features,
    generate_explanation
)

# Import robust QR logic
from core.qr_processing import QRProcessor

app = FastAPI(
    title="PhishX v2: Enhanced QR Phishing Detection System",
    description="Original PhishX Intelligence + Robust QR Extraction",
    version="2.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Paths to ORIGINAL models
SAVED_MODELS_DIR = r"c:\Users\vishn\OneDrive\Desktop\cyber security\PhishX\saved_models"
TRANSFORMER_PATH = os.path.join(SAVED_MODELS_DIR, 'transformer_phishing.pt')
CNN_PATH = os.path.join(SAVED_MODELS_DIR, 'char_cnn_phishing.pt')
GATING_PATH = os.path.join(SAVED_MODELS_DIR, 'gating_network.pt')

# Initialize system components
qr_processor = QRProcessor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global state for models
ensemble = None
transformer_tokenizer = None
char_tokenizer = None

def load_genuine_phishx_models():
    global ensemble, transformer_tokenizer, char_tokenizer
    print("🔄 Loading original PhishX trained models...")
    try:
        transformer_tokenizer = get_transformer_tokenizer()
        char_tokenizer = CharTokenizer()
        
        t_model = URLTransformer()
        c_model = CharCNN(vocab_size=char_tokenizer.vocab_size)
        g_network = UncertaintyAwareGating(feature_dim=8)
        
        # Load weights
        if os.path.exists(TRANSFORMER_PATH):
            t_model.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=device))
            print("✅ Loaded Transformer weights.")
        else:
            print("⚠️ Transformer weights NOT FOUND. Using random weights.")
            
        if os.path.exists(CNN_PATH):
            c_model.load_state_dict(torch.load(CNN_PATH, map_location=device))
            print("✅ Loaded CNN weights.")
        else:
            print("⚠️ CNN weights NOT FOUND. Using random weights.")
            
        if os.path.exists(GATING_PATH):
            g_network.load_state_dict(torch.load(GATING_PATH, map_location=device))
            print("✅ Loaded Adaptive Gating weights.")
        else:
            g_network = None
            print("⚠️ Gating Network NOT FOUND. Falling back to static fusion.")
            
        ensemble = PhishXEnsemble(t_model, c_model, gating_network=g_network)
        print("🚀 PhishX Ensemble Ready.")
    except Exception as e:
        print(f"❌ Error loading original models: {e}")

@app.on_event("startup")
async def startup_event():
    load_genuine_phishx_models()

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class PhishingResponse(BaseModel):
    status: str
    message: str | None = None
    url: str | None = None
    final_url: str | None = None
    prediction: str | None = None
    risk_score: float | None = None
    uncertainty: float | None = None
    alpha: float | None = None
    action: str | None = None
    explanation: list | None = None
    features: dict | None = None

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/scan", response_model=PhishingResponse)
async def scan_qr(file: UploadFile = File(...)):
    """
    Enhanced QR Scan endpoint using original PhishX brain.
    """
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Robust QR Extraction (V2 Improvement)
        qr_results = qr_processor.process_qr(file_path)
        if qr_results["status"] != "success":
            return PhishingResponse(status=qr_results["status"], message=qr_results["message"])
            
        original_url = qr_results["url"]
        
        # 2. Preparation for PhishX Brain
        numeric_features = extract_numeric_features(original_url)
        t_input = transformer_tokenizer(
            original_url, add_special_tokens=True, max_length=128, 
            padding='max_length', truncation=True, return_tensors='pt'
        )
        c_input = char_tokenizer.tokenize(original_url).unsqueeze(0)
        
        # 3. Genuine PhishX Inference
        if ensemble is None:
            raise HTTPException(status_code=503, detail="Models not initialized")
            
        results = ensemble.predict(t_input, c_input, numeric_features=numeric_features, num_passes=10)
        
        # 4. Generate Original Explanations
        explanation = generate_explanation(original_url, results)
        
        # 5. Mapping result to action
        risk = results['p_final']
        uncertainty = results['uncertainty']
        
        action = "Safe"
        if risk > 0.8: action = "Block"
        elif risk > 0.5: action = "Warning"
        
        # Adjust for high uncertainty
        if uncertainty > 0.05:
            action = "Manual Review Required"

        return PhishingResponse(
            status="success",
            url=original_url,
            final_url=original_url, # QR usually gives direct URL
            prediction="Phishing" if risk > 0.5 else "Legitimate",
            risk_score=round(risk, 4),
            uncertainty=round(uncertainty, 4),
            alpha=round(results.get('alpha', 0.5), 4),
            action=action,
            explanation=explanation,
            features=numeric_features
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
