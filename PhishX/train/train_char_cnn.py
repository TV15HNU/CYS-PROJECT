import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.char_cnn_model import CharCNN, CharTokenizer

class CharDataset(Dataset):
    def __init__(self, urls, labels, tokenizer):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, item):
        url = str(self.urls[item])
        label = self.labels[item]
        seq = self.tokenizer.tokenize(url)
        return {
            'input': seq,
            'labels': torch.tensor(label, dtype=torch.float)
        }

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training using: {device}")
    
    # Path Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    data_path = os.path.abspath(os.path.join(base_dir, '..', 'balancedurls\'s', 'KaggleBalancedURLs.csv'))
    if not os.path.exists(data_path):
        data_path = 'balancedurls\'s/KaggleBalancedURLs.csv'
        
    df = pd.read_csv(data_path)
    df = df.sample(n=100000, random_state=42) # Char CNN is faster, can use more data
    df['label'] = df['label'].map({'benign': 0, 'phishing': 1, 'malicious': 1})
    df = df.dropna(subset=['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(df['url'].values, df['label'].values, test_size=0.2, random_state=42)
    
    tokenizer = CharTokenizer()
    train_dataset = CharDataset(X_train, y_train, tokenizer)
    test_dataset = CharDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    model = CharCNN(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(10): # CNNs often need more epochs
        model.train()
        losses = []
        for batch in tqdm(train_loader):
            inputs = batch['input'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        print(f"Epoch {epoch+1} Loss: {np.mean(losses)}")
        
        # Validation
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input'].to(device)
                labels = batch['labels'].numpy()
                outputs = model(inputs)
                # Apply sigmoid manually for accuracy check
                probs = torch.sigmoid(outputs)
                preds.extend((probs.cpu().numpy() > 0.5).astype(int).flatten())
                actuals.extend(labels)
        
        print(f"Validation Accuracy: {accuracy_score(actuals, preds)}")
        
    torch.save(model.state_dict(), os.path.join(save_dir, 'char_cnn_phishing.pt'))
    print(f"Model saved to {os.path.join(save_dir, 'char_cnn_phishing.pt')}")

if __name__ == "__main__":
    train()
