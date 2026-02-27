import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import sys
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import URLTransformer, get_tokenizer

class URLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer, max_len=128):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, item):
        url = str(self.urls[item])
        label = self.labels[item]
        
        encoding = self.tokenizer(
            url,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'url_text': url,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
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
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # ULTIMATE STABILITY MODE
    # Reduced sample to 10k for research validation. 
    # You can increase this to 50k once you confirm it doesn't crash.
    print(f"STABILITY MODE: Sampling 10,000 rows...")
    df = df.sample(n=10000, random_state=42)
    
    df['label'] = df['label'].map({'benign': 0, 'phishing': 1, 'malicious': 1})
    df = df.dropna(subset=['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(df['url'].values, df['label'].values, test_size=0.2, random_state=42)
    
    tokenizer = get_tokenizer()
    train_dataset = URLDataset(X_train, y_train, tokenizer)
    test_dataset = URLDataset(X_test, y_test, tokenizer)
    
    # EXTREME LOW LOAD: Small physical batch + Gradient Accumulation
    # Physical batch = 2 (Tiny GPU footprint)
    # Accumulation = 4 (Total effective batch = 8)
    physical_batch = 2
    accumulation_steps = 4
    
    train_loader = DataLoader(train_dataset, batch_size=physical_batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=physical_batch)
    
    model = URLTransformer().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # MIXED PRECISION (FP16): Reduces heat and memory
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # CHECKPOINTING: Resume if crashed
    checkpoint_path = os.path.join(save_dir, 'transformer_checkpoint.pt')
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("Found checkpoint! Resuming training...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch}")

    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(start_epoch, 3):
        print(f"\n--- Epoch {epoch + 1} / 3 ---")
        model.train()
        losses = []
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)
            
            # Using Mixed Precision Context
            with torch.amp.autocast('cuda') if device.type == 'cuda' else torch.amp.autocast('cpu', enabled=False):
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                # Normalize loss for accumulation
                loss = loss / accumulation_steps
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights after enough accumulation steps
            if (i + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            if i % 200 == 0:
                torch.cuda.empty_cache()
            
        # Clean sync to cool down between epochs
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print("Cooling down GPU... pausing for 5 seconds...")
            import time
            time.sleep(5) 

        # Save Checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, checkpoint_path)
        
        # Validation
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].numpy()
                outputs = model(input_ids, attention_mask)
                # Apply sigmoid manually for accuracy check
                probs = torch.sigmoid(outputs)
                preds.extend((probs.cpu().numpy() > 0.5).astype(int).flatten())
                actuals.extend(labels)
        
        print(f"Validation Accuracy: {accuracy_score(actuals, preds):.4f}")
        
    torch.save(model.state_dict(), os.path.join(save_dir, 'transformer_phishing.pt'))
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print("Success! Final model saved.")

if __name__ == "__main__":
    train()
