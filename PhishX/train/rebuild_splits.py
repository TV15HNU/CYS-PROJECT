import pandas as pd
import numpy as np
import os
import sys
from urllib.parse import urlparse
from sklearn.model_selection import GroupShuffleSplit
import scipy.stats as stats
import re

def get_domain(url):
    try:
        res = urlparse(url)
        return res.netloc if res.netloc else res.path.split('/')[0]
    except:
        return "unknown"

def calculate_entropy(text):
    if not text: return 0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = - sum([p * np.log2(p) for p in prob])
    return entropy

def rebuild_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.abspath(os.path.join(base_dir, '..', "balancedurls's", 'KaggleBalancedURLs.csv'))
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print("--- STEP 1: DATASET INTEGRITY CHECK ---")
    df = pd.read_csv(data_path)
    initial_count = len(df)
    
    # Clean labels
    df['label_num'] = df['label'].map({'benign': 0, 'phishing': 1, 'malicious': 1})
    df = df.dropna(subset=['url', 'label_num'])
    
    # Deduplication
    df = df.drop_duplicates(subset=['url'])
    unique_count = len(df)
    
    # Domain extraction
    df['domain'] = df['url'].apply(get_domain)
    unique_domains = df['domain'].nunique()
    
    print(f"Initial URLs: {initial_count}")
    print(f"Unique URLs: {unique_count} (Removed {initial_count - unique_count})")
    print(f"Unique Domains: {unique_domains}")

    print("\n--- STEP 2: DOMAIN-LEVEL SPLIT ---")
    # Ensuring no domain leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['domain']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    print(f"Train Size: {len(train_df)} ({train_df['label_num'].mean()*100:.1f}% Phishing)")
    print(f"Test Size: {len(test_df)} ({test_df['label_num'].mean()*100:.1f}% Phishing)")
    
    # Verification
    train_domains = set(train_df['domain'])
    test_domains = set(test_df['domain'])
    leakage = train_domains.intersection(test_domains)
    print(f"Domain Leakage: {len(leakage)} domains overlap (Target: 0)")

    print("\n--- STEP 7: DIFFICULTY ANALYSIS ---")
    test_df_copy = test_df.copy()
    test_df_copy['len'] = test_df_copy['url'].apply(len)
    test_df_copy['entropy'] = test_df_copy['url'].apply(calculate_entropy)
    
    print(f"Avg URL Length: {test_df_copy['len'].mean():.2f}")
    print(f"Avg URL Entropy: {test_df_copy['entropy'].mean():.2f}")
    
    # Save splits for evaluation
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    test_df.sample(frac=1, random_state=42).head(10000).to_csv(os.path.join(save_dir, 'scientific_test_set.csv'), index=False)
    print(f"\nExpanded scientific test set (10,000 shuffled samples) saved to {save_dir}")

if __name__ == "__main__":
    rebuild_pipeline()
