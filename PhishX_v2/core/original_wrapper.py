import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer
import re
from urllib.parse import urlparse

# ---------------------------------------------------------
# Original URLTransformer (from PhishX/models/transformer_model.py)
# ---------------------------------------------------------
class URLTransformer(nn.Module):
    def __init__(self, num_classes=1):
        super(URLTransformer, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def get_transformer_tokenizer():
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# ---------------------------------------------------------
# Original CharCNN (from PhishX/models/char_cnn_model.py)
# ---------------------------------------------------------
class CharCNN(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, num_classes=1, max_len=200):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 3, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x1 = F.relu(self.conv1(x))
        x1 = self.pool(x1).squeeze(-1)
        x2 = F.relu(self.conv2(x))
        x2 = self.pool(x2).squeeze(-1)
        x3 = F.relu(self.conv3(x))
        x3 = self.pool(x3).squeeze(-1)
        combined = torch.cat((x1, x2, x3), dim=1)
        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits

class CharTokenizer:
    def __init__(self, max_len=200):
        self.max_len = max_len
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"
        self.char_to_int = {c: i + 1 for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 1

    def tokenize(self, url):
        url = url.lower()
        seq = [self.char_to_int.get(c, 0) for c in url][:self.max_len]
        seq += [0] * (self.max_len - len(seq))
        return torch.tensor(seq).long()

# ---------------------------------------------------------
# Original Gating Network (from PhishX/models/gating_network.py)
# ---------------------------------------------------------
class UncertaintyAwareGating(nn.Module):
    def __init__(self, feature_dim=8):
        super(UncertaintyAwareGating, self).__init__()
        input_dim = 4 + feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, mean_t, var_t, mean_c, var_c, features):
        combined = torch.cat([mean_t, var_t, mean_c, var_c, features], dim=1)
        alpha = self.mlp(combined)
        p_final = alpha * mean_t + (1 - alpha) * mean_c
        systemic_uncertainty = alpha * var_t + (1 - alpha) * var_c
        return p_final, alpha, systemic_uncertainty

# ---------------------------------------------------------
# Original Ensemble (from PhishX/models/ensemble.py)
# ---------------------------------------------------------
class PhishXEnsemble:
    def __init__(self, transformer_model, cnn_model, gating_network=None, transformer_weight=0.7, cnn_weight=0.3):
        self.transformer = transformer_model
        self.cnn = cnn_model
        self.gating = gating_network
        self.t_weight = transformer_weight
        self.c_weight = cnn_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer.to(self.device); self.cnn.to(self.device)
        if self.gating: self.gating.to(self.device)
        self.transformer.eval(); self.cnn.eval()

    def predict(self, transformer_inputs, cnn_inputs, numeric_features=None, num_passes=10):
        self.transformer.train(); self.cnn.train() # MC Dropout active
        t_probs, c_probs = [], []
        with torch.no_grad():
            for _ in range(num_passes):
                t_id = transformer_inputs['input_ids'].to(self.device)
                t_mk = transformer_inputs['attention_mask'].to(self.device)
                t_probs.append(torch.sigmoid(self.transformer(t_id, t_mk)))
                c_in = cnn_inputs.to(self.device)
                c_probs.append(torch.sigmoid(self.cnn(c_in)))
        
        t_stack = torch.stack(t_probs)
        mean_t, var_t = t_stack.mean(dim=0), t_stack.var(dim=0)
        c_stack = torch.stack(c_probs)
        mean_c, var_c = c_stack.mean(dim=0), c_stack.var(dim=0)
        
        if self.gating and numeric_features is not None:
            feat_tensor = torch.tensor([list(numeric_features.values())], dtype=torch.float).to(self.device)
            p_final, alpha, uncertainty = self.gating(mean_t, var_t, mean_c, var_c, feat_tensor)
            return {
                "p_final": p_final.item(), "alpha": alpha.item(), "uncertainty": uncertainty.item(),
                "p_t": mean_t.item(), "p_c": mean_c.item(), "var_t": var_t.item(), "var_c": var_c.item()
            }
        else:
            p_final = (mean_t.item() * self.t_weight) + (mean_c.item() * self.c_weight)
            return {
                "p_final": p_final, "uncertainty": (var_t.item() + var_c.item())/2, "alpha": self.t_weight,
                "p_t": mean_t.item(), "p_c": mean_c.item()
            }

# ---------------------------------------------------------
# Original Utilities (from PhishX/utils/feature_extraction.py)
# ---------------------------------------------------------
def extract_numeric_features(url):
    features = {}
    features['url_length'] = len(url)
    features['dot_count'] = url.count('.')
    features['special_char_count'] = len(re.findall(r'[#@!$%^&*()\-=_+]', url))
    features['digit_count'] = len(re.findall(r'\d', url))
    features['subdomain_count'] = max(0, len(urlparse(url).netloc.split('.')) - 2)
    features['is_https'] = 1 if url.startswith('https') else 0
    features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
    suspicious_keywords = ['login', 'verify', 'update', 'secure', 'bank', 'account', 'signin', 'paypal', 'office']
    features['suspicious_keyword_count'] = sum(1 for kw in suspicious_keywords if kw in url.lower())
    return features

def generate_explanation(url, res_dict):
    explanations = []
    features = extract_numeric_features(url)
    prob = res_dict['p_final']
    uncertainty = res_dict['uncertainty']
    alpha = res_dict.get('alpha', 0.5)
    
    if uncertainty > 0.05:
        explanations.append(f"⚠️ HIGH SYSTEM UNCERTAINTY ({uncertainty:.4f}): The models are producing conflicting results, possibly due to unseen adversarial patterns.")
    if features['has_ip']: explanations.append("Heuristic Alert: Direct IP address detection.")
    if features['suspicious_keyword_count'] > 1: explanations.append(f"Heuristic Alert: Found {features['suspicious_keyword_count']} risky keywords.")
    
    if alpha > 0.8: explanations.append(f"Logic: Decision primarily based on Linguistic Semantics (Weight: {alpha:.2f}).")
    elif alpha < 0.2: explanations.append(f"Logic: Decision primarily based on Structural Layout (Weight: {1-alpha:.2f}).")
        
    if prob > 0.8: explanations.append("Verdict: BLOCK - High-confidence phishing detection.")
    elif prob > 0.5:
        if uncertainty > 0.03: explanations.append("Verdict: WARN - Likely phishing, but with low model confidence. Manual review recommended.")
        else: explanations.append("Verdict: WARN - Suspicious activity detected.")
    else: explanations.append("Verdict: ALLOW - Normal URL patterns observed.")
    return explanations
