import torch
import pandas as pd
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             brier_score_loss)
import psutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import URLTransformer, get_tokenizer
from models.char_cnn_model import CharCNN, CharTokenizer
from models.ensemble import PhishXEnsemble
from models.gating_network import UncertaintyAwareGating
from utils.feature_extraction import extract_numeric_features
from utils.adversarial_attacks import AdversarialURLGenerator

def create_folders(base_path):
    folders = [
        "baseline_comparison",
        "imbalance_evaluation",
        "robustness_analysis",
        "calibration_analysis",
        "cross_validation",
        "computational_analysis"
    ]
    for f in folders:
        os.makedirs(os.path.join(base_path, f), exist_ok=True)
    print(f"Created results directory structure at {base_path}")

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_reliability_diagram(y_true, y_prob, title, save_path, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
        else:
            accuracies.append(0)
            confidences.append((bin_lower + bin_upper) / 2)
            
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.bar(bin_lowers, accuracies, width=1.0/n_bins, align='edge', alpha=0.5, color='blue', edgecolor='black', label='Model')
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_results():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    results_path = os.path.join(base_dir, 'results')
    create_folders(results_path)
    
    # 1. Load Data
    test_path = os.path.join(save_dir, 'scientific_test_set.csv')
    if not os.path.exists(test_path):
        print("Scientific test set not found. Please run rebuild_splits.py first.")
        return
        
    df_test = pd.read_csv(test_path).head(1000) # Limited to 1000 for faster report generation, can increase to 5000
    urls = df_test['url'].values
    y_true = df_test['label_num'].values
    
    # 2. Load Models
    t_tokens = get_tokenizer()
    c_tokens = CharTokenizer()
    t_model = URLTransformer().to(device)
    c_model = CharCNN(vocab_size=c_tokens.vocab_size).to(device)
    g_network = UncertaintyAwareGating(feature_dim=8).to(device)
    
    t_model.load_state_dict(torch.load(os.path.join(save_dir, 'transformer_phishing.pt'), map_location=device))
    c_model.load_state_dict(torch.load(os.path.join(save_dir, 'char_cnn_phishing.pt'), map_location=device))
    g_network.load_state_dict(torch.load(os.path.join(save_dir, 'leak_free_gating_network.pt'), map_location=device))
    
    ensemble_adaptive = PhishXEnsemble(t_model, c_model, gating_network=g_network)
    ensemble_fixed = PhishXEnsemble(t_model, c_model, gating_network=None)
    
    print(f"Starting evaluation on {len(urls)} samples...")
    
    p_t_list, p_c_list, p_f_list, p_a_list, alpha_list, unc_list = [], [], [], [], [], []
    
    for i in tqdm(range(len(urls))):
        url = str(urls[i])
        t_in = t_tokens(url, max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(device)
        c_in = c_tokens.tokenize(url).unsqueeze(0).to(device)
        feats = extract_numeric_features(url)
        
        with torch.no_grad():
            res_a = ensemble_adaptive.predict(t_in, c_in, numeric_features=feats, num_passes=5)
            res_f = ensemble_fixed.predict(t_in, c_in, num_passes=1)
            
            p_t_list.append(res_f['p_t'])
            p_c_list.append(res_f['p_c'])
            p_f_list.append(res_f['p_final'])
            p_a_list.append(res_a['p_final'])
            alpha_list.append(res_a['alpha'])
            unc_list.append(res_a['uncertainty'])
            
    # --- BASELINE COMPARISON ---
    metrics = []
    names = ["Transformer", "CNN", "Fixed Fusion", "Adaptive Gating"]
    probs = [p_t_list, p_c_list, p_f_list, p_a_list]
    
    plt.figure(figsize=(10, 8))
    for name, p in zip(names, probs):
        y_p = (np.array(p) > 0.5).astype(int)
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_p),
            "Precision": precision_score(y_true, y_p),
            "Recall": recall_score(y_true, y_p),
            "F1": f1_score(y_true, y_p),
            "AUC": roc_auc_score(y_true, p),
            "AUPRC": average_precision_score(y_true, p)
        })
        
        # Plot confusion matrices
        plot_confusion_matrix(y_true, y_p, f"Confusion Matrix: {name}", 
                               os.path.join(results_path, "baseline_comparison", f"confusion_matrix_{name.lower().replace(' ', '_')}.png"))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, p):.4f})")
        
    pd.DataFrame(metrics).to_csv(os.path.join(results_path, "baseline_comparison", "metrics_table.csv"), index=False)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_path, "baseline_comparison", "roc_curve_comparison.png"), dpi=300)
    plt.close()
    
    # --- IMBALANCE EVALUATION ---
    imb_metrics = []
    ratio_colors = ["blue", "green", "red"]
    plt.figure(figsize=(10, 8))
    
    for idx, ratio in enumerate([10, 50, 100]):
        mal_idx = np.where(y_true == 1)[0]
        ben_idx = np.where(y_true == 0)[0]
        selected_ben = np.random.choice(ben_idx, min(len(ben_idx), len(mal_idx)*ratio), replace=False)
        combined_idx = np.concatenate([mal_idx, selected_ben])
        
        y_t_imb = y_true[combined_idx]
        p_a_imb = np.array(p_a_list)[combined_idx]
        y_p_imb = (p_a_imb > 0.5).astype(int)
        
        m = {
            "Ratio": f"1:{ratio}",
            "Accuracy": accuracy_score(y_t_imb, y_p_imb),
            "Precision": precision_score(y_t_imb, y_p_imb),
            "Recall": recall_score(y_t_imb, y_p_imb),
            "F1": f1_score(y_t_imb, y_p_imb),
            "AUPRC": average_precision_score(y_t_imb, p_a_imb),
            "FPR": 1 - recall_score(y_t_imb, y_p_imb, pos_label=0)
        }
        imb_metrics.append(m)
        pd.DataFrame([m]).to_csv(os.path.join(results_path, "imbalance_evaluation", f"metrics_1_{ratio}.csv"), index=False)
        
        # PR Curve for this ratio
        prec, rec, _ = precision_recall_curve(y_t_imb, p_a_imb)
        plt.plot(rec, prec, label=f"Ratio 1:{ratio} (AUPRC = {m['AUPRC']:.4f})", color=ratio_colors[idx])
        
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves under Class Imbalance', fontsize=14, fontweight='bold')
    plt.legend()
    plt.savefig(os.path.join(results_path, "imbalance_evaluation", "pr_curve_imbalance.png"), dpi=300)
    plt.close()
    
    # --- ROBUSTNESS ANALYSIS ---
    attack_gen = AdversarialURLGenerator()
    robust_results = []
    alpha_variants = [] # To compare alpha distribution
    
    print("Running robustness evaluation...")
    for i in tqdm(range(len(urls))):
        url = str(urls[i])
        if y_true[i] == 1: # Only attack phishing URLs
            attacked_url = attack_gen.homoglyph_attack(url) # Using one representative attack
            t_in = t_tokens(attacked_url, max_length=128, padding='max_length', truncation=True, return_tensors='pt').to(device)
            c_in = c_tokens.tokenize(attacked_url).unsqueeze(0).to(device)
            feats = extract_numeric_features(attacked_url)
            
            with torch.no_grad():
                res = ensemble_adaptive.predict(t_in, c_in, numeric_features=feats, num_passes=5)
                alpha_variants.append(res['alpha'])
                
    # Plot alpha distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(alpha_list, label='Clean', fill=True, color='blue')
    sns.kdeplot(alpha_variants, label='Under Attack', fill=True, color='red')
    plt.title(r'Gating Bias ($\alpha$) Shift under Adversarial Pressure', fontsize=14, fontweight='bold')
    plt.xlabel(r'Alpha Weight ($\alpha$)', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(results_path, "robustness_analysis", "alpha_distribution_attack.png"), dpi=300)
    plt.close()

    # --- CALIBRATION ---
    plot_reliability_diagram(np.array(y_true), np.array(p_a_list), "Reliability Diagram (Adaptive Gating)",
                              os.path.join(results_path, "calibration_analysis", "reliability_diagram.png"))
                              
    ece_scores = []
    for name, p in zip(names, probs):
        ece_scores.append({
            "Model": name,
            "Brier Score": brier_score_loss(y_true, p)
        })
    pd.DataFrame(ece_scores).to_csv(os.path.join(results_path, "calibration_analysis", "ece_brier_scores.csv"), index=False)
    
    # --- COMPUTATIONAL ---
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()
    for _ in range(10):
        ensemble_adaptive.predict(t_in, c_in, numeric_features=feats, num_passes=1)
    latency_single = (time.time() - start_time) / 10 * 1000
    
    start_time = time.time()
    for _ in range(10):
        ensemble_adaptive.predict(t_in, c_in, numeric_features=feats, num_passes=10)
    latency_mc = (time.time() - start_time) / 10 * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    comp_data = {
        "Metric": ["Inference Latency (ms)", "Throughput (req/s)", "Memory Overhead (MB)"],
        "Single Pass": [latency_single, 1000/latency_single, 0],
        "MC (10x)": [latency_mc, 1000/latency_mc, mem_after - mem_before]
    }
    pd.DataFrame(comp_data).to_csv(os.path.join(results_path, "computational_analysis", "memory_usage.csv"), index=False)
    
    # Export RAW Predictions
    raw_df = pd.DataFrame({
        "url": urls,
        "true_label": y_true,
        "p_transformer": p_t_list,
        "p_cnn": p_c_list,
        "p_final": p_a_list,
        "alpha": alpha_list,
        "uncertainty": unc_list
    })
    raw_df.to_csv(os.path.join(results_path, "raw_predictions_dump.csv"), index=False)
    
    # SUMMARY REPORT
    with open(os.path.join(results_path, "summary_report.txt"), "w") as f:
        f.write("PhishX Research Experiment Summary\n")
        f.write("==================================\n\n")
        f.write(f"Generated on: {time.ctime()}\n")
        f.write(f"Test Set: scientific_test_set.csv ({len(urls)} samples)\n")
        f.write(f"Leakage Protection: Domain-level splitting used.\n\n")
        f.write("Folders:\n")
        f.write("- baseline_comparison: Overall precision/recall/F1 vs baselines.\n")
        f.write("- imbalance_evaluation: Robustness to realistic benign/malicious ratios.\n")
        f.write("- robustness_analysis: Evidence of model reliability shifting under attack.\n")
        f.write("- calibration_analysis: Reliability diagrams and Brier scores.\n")
        f.write("- computational_analysis: Performance tradeoffs for Monte Carlo inference.\n\n")
        f.write("Regeneration: Run 'python PhishX/train/generate_publication_results.py'\n")

    print(f"\nAll results successfully organized in {results_path}")

if __name__ == "__main__":
    generate_results()
