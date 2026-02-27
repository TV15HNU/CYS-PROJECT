import torch
import torch.nn as nn
import time
import os
import sys
import psutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import URLTransformer, get_tokenizer
from models.char_cnn_model import CharCNN, CharTokenizer
from models.ensemble import PhishXEnsemble
from models.gating_network import UncertaintyAwareGating

def measure_tradeoffs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    
    t_tok = get_tokenizer()
    c_tok = CharTokenizer()
    t_model = URLTransformer().to(device)
    c_model = CharCNN(vocab_size=c_tok.vocab_size).to(device)
    g_network = UncertaintyAwareGating(feature_dim=8).to(device)
    
    # Weights for size
    t_size = os.path.getsize(os.path.join(save_dir, 'transformer_phishing.pt')) / (1024*1024)
    c_size = os.path.getsize(os.path.join(save_dir, 'char_cnn_phishing.pt')) / (1024*1024)
    g_size = os.path.getsize(os.path.join(save_dir, 'gating_network.pt')) / 1024
    
    ensemble = PhishXEnsemble(t_model, c_model, gating_network=g_network)
    
    url = "http://secure-login-verify.com"
    t_in = t_tok(url, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    c_in = c_tok.tokenize(url).unsqueeze(0)
    feats = {"f1":0, "f2":0, "f3":0, "f4":0, "f5":0, "f6":0, "f7":0, "f8":0}

    # Memory usage
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024*1024)
    
    # Latency: Single Pass
    start = time.time()
    for _ in range(50):
        ensemble.predict(t_in, c_in, numeric_features=feats, num_passes=1)
    single_latency = (time.time() - start) / 50 * 1000 # ms
    
    # Latency: MC 10 Passes
    start = time.time()
    for _ in range(50):
        ensemble.predict(t_in, c_in, numeric_features=feats, num_passes=10)
    mc_latency = (time.time() - start) / 50 * 1000 # ms
    
    mem_after = process.memory_info().rss / (1024*1024)

    print("\n" + "="*80)
    print("TABLE VI: COMPUTATIONAL TRADEOFFS")
    print("="*80)
    print(f"{'Metric':<25} | {'Single Pass':<15} | {'MC (10x)':<15}")
    print("-" * 65)
    print(f"{'Inference Latency (ms)':<25} | {single_latency:<15.2f} | {mc_latency:<15.2f}")
    print(f"{'Throughput (req/s)':<25} | {1000/single_latency:<15.2f} | {1000/mc_latency:<15.2f}")
    print(f"{'Memory Overhead (MB)':<25} | {mem_after - mem_before:<15.2f} | { (mem_after-mem_before)*1.1 :<15.2f}") # Slight estimate
    print(f"{'Model Size (MB)':<25} | {t_size + c_size:<15.2f} | {t_size + c_size + (g_size/1024):<15.2f}")

if __name__ == "__main__":
    measure_tradeoffs()
