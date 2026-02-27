import torch
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import URLTransformer, get_tokenizer
from models.char_cnn_model import CharCNN, CharTokenizer
from utils.feature_extraction import extract_numeric_features

def get_domain(url):
    try: return urlparse(url).netloc
    except: return "unk"

def extract_for_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    data_path = os.path.abspath(os.path.join(base_dir, '..', "balancedurls's", 'KaggleBalancedURLs.csv'))
    
    df = pd.read_csv(data_path).drop_duplicates(subset=['url'])
    df['label_num'] = df['label'].map({'benign': 0, 'phishing': 1, 'malicious': 1})
    df['domain'] = df['url'].apply(lambda x: str(x).split('/')[2] if '://' in str(x) else str(x).split('/')[0])
    
    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(gss.split(df, groups=df['domain']))
    train_df = df.iloc[train_idx].sample(15000, random_state=42) # 15k for fast gating tune
    
    # Load Backbone Models
    t_model = URLTransformer().to(device)
    c_tokenizer = CharTokenizer()
    c_model = CharCNN(vocab_size=c_tokenizer.vocab_size).to(device)
    
    t_model.load_state_dict(torch.load(os.path.join(save_dir, 'transformer_phishing.pt'), map_location=device))
    c_model.load_state_dict(torch.load(os.path.join(save_dir, 'char_cnn_phishing.pt'), map_location=device))
    
    t_model.train() # Enable dropout for uncertainty
    c_model.train()
    t_tokenizer = get_tokenizer()

    urls = train_df['url'].values
    labels = train_df['label_num'].values
    extracted_data = []

    print("Extracting Leak-Free Bayesian Features (15k samples)...")
    with torch.no_grad():
        for i in tqdm(range(len(urls))):
            url = str(urls[i])
            label = labels[i]
            
            # 1. Bayesian inference (10 passes)
            t_pass_probs, c_pass_probs = [], []
            for _ in range(5): # Reduced to 5 for training speed
                t_input = t_tokenizer(url, max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(device)
                t_pass_probs.append(torch.sigmoid(t_model(t_input['input_ids'], t_input['attention_mask'])).item())
                
                c_input = c_tokenizer.tokenize(url).unsqueeze(0).to(device)
                c_pass_probs.append(torch.sigmoid(c_model(c_input)).item())
            
            mean_t, var_t = np.mean(t_pass_probs), np.var(t_pass_probs)
            mean_c, var_c = np.mean(c_pass_probs), np.var(c_pass_probs)
            feats = list(extract_numeric_features(url).values())
            extracted_data.append([mean_t, var_t, mean_c, var_c] + feats + [label])

    df_output = pd.DataFrame(extracted_data, columns=['mean_t', 'var_t', 'mean_c', 'var_c'] + [f'f{j}' for j in range(8)] + ['label'])
    df_output.to_csv(os.path.join(save_dir, 'leak_free_gating_features.csv'), index=False)
    print("\nExtraction complete.")

if __name__ == "__main__":
    extract_for_training()
