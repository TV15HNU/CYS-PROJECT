import torch
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import URLTransformer, get_tokenizer
from models.char_cnn_model import CharCNN, CharTokenizer
from utils.feature_extraction import extract_numeric_features

def extract():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extracting features using: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    
    # Load Models
    t_model = URLTransformer().to(device)
    c_tokenizer = CharTokenizer()
    c_model = CharCNN(vocab_size=c_tokenizer.vocab_size).to(device)
    
    t_model.load_state_dict(torch.load(os.path.join(save_dir, 'transformer_phishing.pt'), map_location=device))
    c_model.load_state_dict(torch.load(os.path.join(save_dir, 'char_cnn_phishing.pt'), map_location=device))
    
    t_model.eval()
    c_model.eval()
    t_tokenizer = get_tokenizer()

    # Load Data
    data_path = os.path.abspath(os.path.join(base_dir, '..', 'balancedurls\'s', 'KaggleBalancedURLs.csv'))
    df = pd.read_csv(data_path).sample(10000, random_state=42) # Limited to 10k for safety
    df['label'] = df['label'].map({'benign': 0, 'phishing': 1, 'malicious': 1})
    urls = df['url'].values
    labels = df['label'].values

    extracted_data = []

    print("Running one-time feature extraction with MC Dropout (10 passes)...")
    # Enable Dropout during evaluation
    t_model.train()
    c_model.train()

    with torch.no_grad():
        for i in tqdm(range(len(urls))):
            url = str(urls[i])
            label = labels[i]
            
            # 1. Bayesian inference (10 passes)
            t_pass_probs, c_pass_probs = [], []
            for _ in range(10):
                t_input = t_tokenizer(url, max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(device)
                t_pass_probs.append(torch.sigmoid(t_model(t_input['input_ids'], t_input['attention_mask'])).item())
                
                c_input = c_tokenizer.tokenize(url).unsqueeze(0).to(device)
                c_pass_probs.append(torch.sigmoid(c_model(c_input)).item())
            
            mean_t, var_t = np.mean(t_pass_probs), np.var(t_pass_probs)
            mean_c, var_c = np.mean(c_pass_probs), np.var(c_pass_probs)
            
            # 2. Numeric Features
            feats = list(extract_numeric_features(url).values())
            
            extracted_data.append([mean_t, var_t, mean_c, var_c] + feats + [label])

    # Save to file
    cols = ['mean_t', 'var_t', 'mean_c', 'var_c'] + [f'f{j}' for j in range(8)] + ['label']
    df_output = pd.DataFrame(extracted_data, columns=cols)
    df_output.to_csv(os.path.join(save_dir, 'gating_features.csv'), index=False)
    print(f"\nSuccess! Features extracted and saved to {os.path.join(save_dir, 'gating_features.csv')}")

if __name__ == "__main__":
    extract()
