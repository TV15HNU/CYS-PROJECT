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

def train_leak_free():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    data_path = os.path.join(save_dir, 'leak_free_gating_features.csv')
    
    if not os.path.exists(data_path):
        print("Error: leak_free_gating_features.csv not found.")
        return

    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float)
    
    g_network = UncertaintyAwareGating(feature_dim=8)
    optimizer = torch.optim.Adam(g_network.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    
    print("Training Leak-Free Gating...")
    for epoch in range(50):
        g_network.train()
        
        # Inputs: mean_t, var_t, mean_c, var_c, f1-f8
        p_final_prob, _, _ = g_network(X_train[:, 0:1], X_train[:, 1:2], X_train[:, 2:3], X_train[:, 3:4], X_train[:, 4:])
        
        eps = 1e-7
        p_final_prob = torch.clamp(p_final_prob, min=eps, max=1-eps)
        p_final_logits = torch.log(p_final_prob / (1 - p_final_prob))
        
        loss = loss_fn(p_final_logits, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Stats
    g_network.eval()
    with torch.no_grad():
        p_val_p, _, _ = g_network(X_val[:, 0:1], X_val[:, 1:2], X_val[:, 2:3], X_val[:, 3:4], X_val[:, 4:])
        acc = accuracy_score(y_val, (p_val_p.numpy() > 0.5))
        print(f"Validation Accuracy: {acc:.4f}")

    torch.save(g_network.state_dict(), os.path.join(save_dir, 'leak_free_gating_network.pt'))
    print(f"Saved to {os.path.join(save_dir, 'leak_free_gating_network.pt')}")

if __name__ == "__main__":
    train_leak_free()
