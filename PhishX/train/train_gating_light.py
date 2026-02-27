import torch
import torch.nn as nn
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gating_network import UncertaintyAwareGating

def train_lightweight():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    data_path = os.path.join(save_dir, 'gating_features.csv')
    
    if not os.path.exists(data_path):
        print("Error: gating_features.csv not found. Run extract_gating_data.py first!")
        return

    # Load pre-computed features
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Move to tensors
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float)
    
    # Initialize Gating Network (The tiny MLP)
    g_network = UncertaintyAwareGating(feature_dim=8)
    optimizer = torch.optim.Adam(g_network.parameters(), lr=0.01) # Higher LR for fast convergence
    loss_fn = nn.BCEWithLogitsLoss()
    
    print("Training Gating Network (Lightweight Mode)...")
    for epoch in range(50): # We can run many epochs because it's instant
        g_network.train()
        
        # Bayesian Inputs: mean_t, var_t, mean_c, var_c
        mean_t = X_train[:, 0:1]
        var_t  = X_train[:, 1:2]
        mean_c = X_train[:, 2:3]
        var_c  = X_train[:, 3:4]
        feats  = X_train[:, 4:]
        
        p_final_prob, alpha, _ = g_network(mean_t, var_t, mean_c, var_c, feats)
        
        # Logit conversion for stable loss
        eps = 1e-7
        p_final_prob = torch.clamp(p_final_prob, min=eps, max=1-eps)
        p_final_logits = torch.log(p_final_prob / (1 - p_final_prob))
        
        loss = loss_fn(p_final_logits, y_train)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Final Validation
    g_network.eval()
    with torch.no_grad():
        m_t, v_t, m_c, v_c = X_test[:, 0:1], X_test[:, 1:2], X_test[:, 2:3], X_test[:, 3:4]
        p_final_prob, _, _ = g_network(m_t, v_t, m_c, v_c, X_test[:, 4:])
        preds = (p_final_prob.numpy() > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"\nFinal Gating Accuracy: {acc:.4f}")

    torch.save(g_network.state_dict(), os.path.join(save_dir, 'gating_network.pt'))
    print(f"Gating Network Saved to {os.path.join(save_dir, 'gating_network.pt')}")

if __name__ == "__main__":
    train_lightweight()
