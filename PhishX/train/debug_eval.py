import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from models.transformer_model import URLTransformer, get_tokenizer
    from models.char_cnn_model import CharCNN, CharTokenizer
    from models.ensemble import PhishXEnsemble
    from models.gating_network import AdaptiveGatingNetwork
    print("Imports Success!")
except Exception as e:
    print(f"Import Error: {e}")
