import torch
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import URLTransformer, get_tokenizer
from models.char_cnn_model import CharCNN, CharTokenizer
from models.ensemble import PhishXEnsemble
from models.gating_network import UncertaintyAwareGating
from utils.feature_extraction import extract_numeric_features
from utils.adversarial_attacks import AdversarialURLGenerator

def evaluate_robustness():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Robustness Evaluation on: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    
    # 1. Load All Models
    t_tokenizer = get_tokenizer()
    c_tokenizer = CharTokenizer()
    
    t_model = URLTransformer().to(device)
    c_model = CharCNN(vocab_size=c_tokenizer.vocab_size).to(device)
    g_network = UncertaintyAwareGating(feature_dim=8).to(device)
    
    t_model.load_state_dict(torch.load(os.path.join(save_dir, 'transformer_phishing.pt'), map_location=device))
    c_model.load_state_dict(torch.load(os.path.join(save_dir, 'char_cnn_phishing.pt'), map_location=device))
    g_network.load_state_dict(torch.load(os.path.join(save_dir, 'gating_network.pt'), map_location=device))
    
    ensemble_adaptive = PhishXEnsemble(t_model, c_model, gating_network=g_network)
    ensemble_fixed = PhishXEnsemble(t_model, c_model, gating_network=None) # Uses 0.7/0.3 fallback
    
    # 2. Load Evaluation Data (Malicious URLs only for robustness testing)
    data_path = os.path.abspath(os.path.join(base_dir, '..', 'balancedurls\'s', 'KaggleBalancedURLs.csv'))
    df = pd.read_csv(data_path)
    df['label_num'] = df['label'].map({'benign': 0, 'phishing': 1, 'malicious': 1})
    
    # We test on malicious samples to see if the attacks hide them successfully
    eval_df = df[df['label_num'] == 1].sample(100, random_state=42)
    urls = eval_df['url'].values
    
    attack_gen = AdversarialURLGenerator()
    attacks = {
        "Clean": lambda x: x,
        "Homoglyph": attack_gen.homoglyph_attack,
        "Typosquatting": attack_gen.typosquatting_attack,
        "Subdomain Flooding": attack_gen.subdomain_flooding_attack,
        "TLD Squatting": attack_gen.tld_squatting_attack,
        "Prefix Injection": attack_gen.prefix_injection_attack
    }
    
    results = []
    alpha_distribution = {}

    for attack_name, attack_func in attacks.items():
        print(f"\nEvaluating Attack: {attack_name}")
        t_preds, c_preds, fixed_preds, adaptive_preds = [], [], [], []
        alphas = []
        
        for url in tqdm(urls):
            adv_url = attack_func(url)
            
            # Prepare inputs
            t_input = t_tokenizer(adv_url, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            c_input = c_tokenizer.tokenize(adv_url).unsqueeze(0)
            feats = extract_numeric_features(adv_url)
            
            # 1. Adaptive Prediction (returns: final_prob, t_prob, c_prob, alpha)
            prob_a, pt, pc, alpha = ensemble_adaptive.predict(t_input, c_input, numeric_features=feats)
            adaptive_preds.append(1 if prob_a > 0.5 else 0)
            alphas.append(alpha)
            
            # 2. Fixed Prediction
            prob_f, _, _, _ = ensemble_fixed.predict(t_input, c_input)
            fixed_preds.append(1 if prob_f > 0.5 else 0)
            
            # 3. Individual Models
            t_preds.append(1 if pt > 0.5 else 0)
            c_preds.append(1 if pc > 0.5 else 0)

        # Calculate Accuracy (since we only use malicious samples, Acc = % correctly identified as malicious)
        metrics = {
            "Attack": attack_name,
            "Transformer": np.mean(t_preds),
            "CNN": np.mean(c_preds),
            "Fixed": np.mean(fixed_preds),
            "Adaptive": np.mean(adaptive_preds),
            "Avg_Alpha": np.mean(alphas)
        }
        results.append(metrics)
        alpha_distribution[attack_name] = alphas

    # 3. Output Table
    res_df = pd.DataFrame(results)
    res_df["Delta_Imp"] = res_df["Adaptive"] - res_df["Fixed"]
    print("\n" + "="*80)
    print("ADVERSARIAL ROBUSTNESS RESULTS")
    print("="*80)
    print(res_df.to_string(index=False))
    
    # 4. Persistence
    res_df.to_csv(os.path.join(save_dir, 'robustness_results.csv'), index=False)
    
    # 5. Visualizations
    plt.figure(figsize=(10, 6))
    x = np.arange(len(res_df["Attack"]))
    width = 0.2
    
    plt.bar(x - width*1.5, res_df["Transformer"], width, label='Transformer', color='#ff9999')
    plt.bar(x - width*0.5, res_df["CNN"], width, label='CNN', color='#66b3ff')
    plt.bar(x + width*0.5, res_df["Fixed"], width, label='Fixed Fusion', color='#99ff99')
    plt.bar(x + width*1.5, res_df["Adaptive"], width, label='Adaptive Gating', color='#ffcc99')
    
    plt.xlabel('Attack Type')
    plt.ylabel('Malicious Detection Rate')
    plt.title('Robustness Under Adversarial Attacks')
    plt.xticks(x, res_df["Attack"], rotation=15)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'robustness_chart.png'))
    print(f"\nCharts saved to {save_dir}")

if __name__ == "__main__":
    evaluate_robustness()
