import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import URLTransformer, get_tokenizer
from models.char_cnn_model import CharCNN, CharTokenizer
from models.gating_network import UncertaintyAwareGating
from utils.feature_extraction import extract_numeric_features

class GatingDataset(Dataset):
    def __init__(self, urls, labels, t_tokenizer, c_tokenizer):
        self.urls = urls
        self.labels = labels
        self.t_tokenizer = t_tokenizer
        self.c_tokenizer = c_tokenizer

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = str(self.urls[idx])
        label = self.labels[idx]
        
        # Transformer inputs
        t_input = self.t_tokenizer(
            url, add_special_tokens=True, max_length=128, 
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        # CNN inputs
        c_input = self.c_tokenizer.tokenize(url)
        
        # Numeric features
        features = list(extract_numeric_features(url).values())
        
        return {
            't_ids': t_input['input_ids'].squeeze(0),
            't_mask': t_input['attention_mask'].squeeze(0),
            'c_input': c_input,
            'features': torch.tensor(features, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.float)
        }

def train_gating():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Gating Network on: {device}")
    
    # Path Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    
    # Load Backbone Models
    t_tokenizer = get_tokenizer()
    c_tokenizer = CharTokenizer()
    
    t_model = URLTransformer().to(device)
    c_model = CharCNN(vocab_size=c_tokenizer.vocab_size).to(device)
    
    t_model.load_state_dict(torch.load(os.path.join(save_dir, 'transformer_phishing.pt'), map_location=device))
    c_model.load_state_dict(torch.load(os.path.join(save_dir, 'char_cnn_phishing.pt'), map_location=device))
    
    # FREEZE BACKBONES
    for param in t_model.parameters(): param.requires_grad = False
    for param in c_model.parameters(): param.requires_grad = False
    t_model.eval()
    c_model.eval()
    
    # Initialize Gating Network
    g_network = AdaptiveGatingNetwork(feature_dim=8).to(device)
    optimizer = torch.optim.Adam(g_network.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss() # Numerically stable and autocast safe
    
    # Load Data
    data_path = os.path.abspath(os.path.join(base_dir, '..', 'balancedurls\'s', 'KaggleBalancedURLs.csv'))
    df = pd.read_csv(data_path).sample(20000, random_state=42) # Use 20k for gating tune
    df['label'] = df['label'].map({'benign': 0, 'phishing': 1, 'malicious': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(df['url'].values, df['label'].values, test_size=0.2)
    
    print("Loading datasets...")
    train_loader = DataLoader(GatingDataset(X_train, y_train, t_tokenizer, c_tokenizer), batch_size=8, shuffle=True)
    test_loader = DataLoader(GatingDataset(X_test, y_test, t_tokenizer, c_tokenizer), batch_size=8)
    
    print(f"Starting training on {len(X_train)} samples...")
    # Training Loop
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    for epoch in range(5):
        g_network.train()
        total_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            t_ids = batch['t_ids'].to(device)
            t_mask = batch['t_mask'].to(device)
            c_input = batch['c_input'].to(device)
            feats = batch['features'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            # Using Mixed Precision Context
            with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.amp.autocast('cpu', enabled=False):
                with torch.no_grad():
                    # Backbones already return logits, so we apply sigmoid here
                    # as input to the GATING network (which needs probabilities)
                    p_t = torch.sigmoid(t_model(t_ids, t_mask))
                    p_c = torch.sigmoid(c_model(c_input))
                
                # The GATING model forward should ideally return logits for P_final
                # But our current formula P = alpha*Pt + (1-alpha)*Pc works on probabilities.
                # To make BCEWithLogitsLoss work, we need to pass it RAW LOGITS.
                # Since p_t and p_c are probabilities, p_final is a probability.
                # We convert it back to a logit carefully using logit function.
                p_final_prob, alpha = g_network(p_t, p_c, feats)
                
                # Avoid log(0) or log(1) issues
                eps = 1e-7
                p_final_prob = torch.clamp(p_final_prob, min=eps, max=1-eps)
                p_final_logits = torch.log(p_final_prob / (1 - p_final_prob))
                
                loss = loss_fn(p_final_logits, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            total_loss += loss.item()
            
            if i % 100 == 0:
                torch.cuda.empty_cache()
            
        print(f"Loss: {total_loss/len(train_loader):.4f}")
        
        # Validation
        g_network.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for batch in test_loader:
                t_ids = batch['t_ids'].to(device)
                t_mask = batch['t_mask'].to(device)
                c_input = batch['c_input'].to(device)
                feats = batch['features'].to(device)
                
                p_t = torch.sigmoid(t_model(t_ids, t_mask))
                p_c = torch.sigmoid(c_model(c_input))
                p_final_prob, _ = g_network(p_t, p_c, feats)
                
                # Here p_final_prob is already a probability [0, 1]
                preds.extend((p_final_prob.cpu().numpy() > 0.5).astype(int).flatten())
                actuals.extend(batch['label'].numpy())
        
        print(f"Validation Accuracy: {accuracy_score(actuals, preds):.4f}")

        # Cool down laptop between epochs
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            import time
            print("Cooling down GPU... pausing for 5 seconds...")
            time.sleep(5)

    torch.save(g_network.state_dict(), os.path.join(save_dir, 'gating_network.pt'))
    print("Gating Network Saved!")

if __name__ == "__main__":
    train_gating()
