import torch
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, brier_score_loss)
from scipy.stats import chi2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import URLTransformer, get_tokenizer
from models.char_cnn_model import CharCNN, CharTokenizer
from models.ensemble import PhishXEnsemble
from models.gating_network import UncertaintyAwareGating
from utils.feature_extraction import extract_numeric_features

def run_scientific_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    
    # Load test set (5000 clean samples, no domain leakage)
    test_path = os.path.join(save_dir, 'scientific_test_set.csv')
    df_test = pd.read_csv(test_path).sample(frac=1, random_state=42).head(5000)
    
    # Load models
    t_tok = get_tokenizer()
    c_tok = CharTokenizer()
    t_model = URLTransformer().to(device)
    c_model = CharCNN(vocab_size=c_tok.vocab_size).to(device)
    g_network = UncertaintyAwareGating(feature_dim=8).to(device)
    
    t_model.load_state_dict(torch.load(os.path.join(save_dir, 'transformer_phishing.pt'), map_location=device))
    c_model.load_state_dict(torch.load(os.path.join(save_dir, 'char_cnn_phishing.pt'), map_location=device))
    # CRITICAL: Use the leak-free network
    g_network.load_state_dict(torch.load(os.path.join(save_dir, 'leak_free_gating_network.pt'), map_location=device))
    
    ensemble_adaptive = PhishXEnsemble(t_model, c_model, gating_network=g_network)
    ensemble_fixed = PhishXEnsemble(t_model, c_model, gating_network=None)
    
    urls = df_test['url'].values
    y_true = df_test['label_num'].values
    
    p_t_list, p_c_list, p_f_list, p_a_list = [], [], [], []
    
    print(f"Evaluating {len(urls)} leakage-free samples...")
    for i in tqdm(range(len(urls))):
        url = str(urls[i])
        t_in = t_tok(url, max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(device)
        c_in = c_tok.tokenize(url).unsqueeze(0).to(device)
        feats = extract_numeric_features(url)
        
        with torch.no_grad():
            res_a = ensemble_adaptive.predict(t_in, c_in, numeric_features=feats, num_passes=5)
            res_f = ensemble_fixed.predict(t_in, c_in, num_passes=1)
            
            p_t_list.append(res_f['p_t'])
            p_c_list.append(res_f['p_c'])
            p_f_list.append(res_f['p_final'])
            p_a_list.append(res_a['p_final'])

    def get_row(y_t, y_p_prob, name):
        y_p = (np.array(y_p_prob) > 0.5).astype(int)
        return {
            "Model": name,
            "Acc": accuracy_score(y_t, y_p),
            "Prec": precision_score(y_t, y_p),
            "Rec": recall_score(y_t, y_p),
            "F1": f1_score(y_t, y_p),
            "AUC": roc_auc_score(y_t, y_p_prob),
            "AUPRC": average_precision_score(y_t, y_p_prob)
        }

    # Table 1: Final Comparison
    results = [
        get_row(y_true, p_t_list, "Transformer-only"),
        get_row(y_true, p_c_list, "CNN-only"),
        get_row(y_true, p_f_list, "Fixed Fusion"),
        get_row(y_true, p_a_list, "Adaptive Gating (Ours)")
    ]
    df_results = pd.DataFrame(results)

    mal_idx = np.where(y_true == 1)[0]
    ben_idx = np.where(y_true == 0)[0]
    print(f"Test Set stats: Malicious={len(mal_idx)}, Benign={len(ben_idx)}")

    # Imbalanced Tests
    imb_results = []
    for ratio in [10, 50, 100]:
        if len(mal_idx) == 0 or len(ben_idx) == 0:
            print(f"Skipping 1:{ratio} (Empty class)")
            continue
            
        selected_ben_count = min(len(ben_idx), len(mal_idx)*ratio)
        selected_ben = np.random.choice(ben_idx, selected_ben_count, replace=False)
        idx = np.concatenate([mal_idx, selected_ben])
        
        y_t_imb = y_true[idx]
        p_a_imb = np.array(p_a_list)[idx]
        y_p_imb = (p_a_imb > 0.5).astype(int)
        
        imb_results.append({
            "Ratio": f"1:{ratio}",
            "Prec": precision_score(y_t_imb, y_p_imb),
            "Rec": recall_score(y_t_imb, y_p_imb),
            "AUPRC": average_precision_score(y_t_imb, p_a_imb),
            "FPR": 1 - recall_score(y_t_imb, y_p_imb, pos_label=0)
        })

    # McNemar's
    y_p_f = (np.array(p_f_list) > 0.5).astype(int)
    y_p_a = (np.array(p_a_list) > 0.5).astype(int)
    
    # Contingency Table for McNemar
    c11 = np.sum((y_p_f == y_true) & (y_p_a == y_true))
    c12 = np.sum((y_p_f == y_true) & (y_p_a != y_true))
    c21 = np.sum((y_p_f != y_true) & (y_p_a == y_true))
    c22 = np.sum((y_p_f != y_true) & (y_p_a != y_true))
    
    # Manual McNemar Statistics
    stat = ((abs(c12 - c21) - 1)**2) / (c12 + c21)
    p_val = 1 - chi2.cdf(stat, 1)

    # Output
    print("\n" + "="*80)
    print("SCIENTIFIC EVALUATION RESULTS (DOMAIN-LEAKAGE FREE)")
    print("="*80)
    print(df_results.to_string(index=False))
    
    print("\n" + "="*80)
    print("REALISTIC TRAFFIC SIMULATION (IMBALANCED)")
    print("="*80)
    print(pd.DataFrame(imb_results).to_string(index=False))
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE (McNemar's)")
    print("="*80)
    print(f"chi-square: {stat:.4f}, p-value: {p_val:.10f}")
    
    print("\nCONFUSION MATRIX (Adaptive Gating):")
    cm = confusion_matrix(y_true, y_p_a)
    print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")

if __name__ == "__main__":
    run_scientific_eval()
