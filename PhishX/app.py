from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import os
import sys

from models.transformer_model import URLTransformer, get_tokenizer
from models.char_cnn_model import CharCNN, CharTokenizer
from models.ensemble import PhishXEnsemble
from utils.qr_scanner import scan_qr_from_image
from utils.feature_extraction import extract_numeric_features, generate_explanation
from utils.crawler import crawl_url

app = FastAPI(title="PhishX - Advanced Phishing Detection")

from models.gating_network import UncertaintyAwareGating

# Models paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_PATH = os.path.join(BASE_DIR, 'saved_models', 'transformer_phishing.pt')
CNN_PATH = os.path.join(BASE_DIR, 'saved_models', 'char_cnn_phishing.pt')
GATING_PATH = os.path.join(BASE_DIR, 'saved_models', 'gating_network.pt')

# Global model state
ensemble = None
transformer_tokenizer = None
char_tokenizer = None

def load_models():
    global ensemble, transformer_tokenizer, char_tokenizer
    try:
        transformer_tokenizer = get_tokenizer()
        char_tokenizer = CharTokenizer()
        
        t_model = URLTransformer()
        c_model = CharCNN(vocab_size=char_tokenizer.vocab_size)
        g_network = UncertaintyAwareGating(feature_dim=8)
        
        if os.path.exists(TRANSFORMER_PATH):
            t_model.load_state_dict(torch.load(TRANSFORMER_PATH, map_location='cpu'))
            print("Loaded Transformer weights.")
            
        if os.path.exists(CNN_PATH):
            c_model.load_state_dict(torch.load(CNN_PATH, map_location='cpu'))
            print("Loaded CNN weights.")
            
        if os.path.exists(GATING_PATH):
            g_network.load_state_dict(torch.load(GATING_PATH, map_location='cpu'))
            print("Loaded Adaptive Gating weights.")
        else:
            g_network = None # Fallback to static weights in ensemble
            print("Using fallback static weights (0.7/0.3).")
            
        ensemble = PhishXEnsemble(t_model, c_model, gating_network=g_network)
        print("Ensemble ready.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.on_event("startup")
async def startup_event():
    load_models()

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_qr(request: Request):
    data = await request.json()
    base64_img = data.get("image")
    
    if not base64_img:
        raise HTTPException(status_code=400, detail="No image data provided")
        
    # 1. Scan QR
    url = scan_qr_from_image(base64_img)
    if not url:
        return {"success": False, "message": "No QR code detected"}
        
    # 2. Extract numeric features
    features = extract_numeric_features(url)
    
    # 3. Model Prediction
    t_input = transformer_tokenizer(
        url, add_special_tokens=True, max_length=128, 
        padding='max_length', truncation=True, return_tensors='pt'
    )
    c_input = char_tokenizer.tokenize(url).unsqueeze(0)
    
    # NEW: Bayesian Inference with 10 stochastic passes
    results = ensemble.predict(t_input, c_input, numeric_features=features, num_passes=10)
    
    # Extract values for response
    prob = results['p_final']
    uncertainty = results['uncertainty']
    alpha = results['alpha']
    
    # 4. Crawling (Optional Advanced)
    crawl_results = crawl_url(url)
    
    # 5. Generate Explanation (Passing the whole results dict)
    explanation = generate_explanation(url, results)
    
    return {
        "success": True,
        "url": url,
        "prediction": "Phishing" if prob > 0.5 else "Legitimate",
        "risk_score": round(prob, 4),
        "uncertainty_score": round(uncertainty, 4),
        "adaptive_weight_alpha": round(alpha, 4),
        "transformer_confidence": round(results['p_t'], 4),
        "cnn_confidence": round(results['p_c'], 4),
        "explanation": explanation,
        "crawl_data": crawl_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
