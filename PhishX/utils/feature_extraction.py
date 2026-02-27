import re
from urllib.parse import urlparse

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
    
    # 1. Uncertainty Analysis (Trustworthiness)
    if uncertainty > 0.05:
        explanations.append(f"⚠️ HIGH SYSTEM UNCERTAINTY ({uncertainty:.4f}): The models are producing conflicting results, possibly due to unseen adversarial patterns.")
    
    # 2. Heuristics
    if features['has_ip']:
        explanations.append("Heuristic Alert: Direct IP address detection.")
    if features['suspicious_keyword_count'] > 1:
        explanations.append(f"Heuristic Alert: Found {features['suspicious_keyword_count']} risky keywords.")
        
    # 3. Model Reliance (Gating)
    if alpha > 0.8:
        explanations.append(f"Logic: Decision primarily based on Linguistic Semantics (Weight: {alpha:.2f}).")
    elif alpha < 0.2:
        explanations.append(f"Logic: Decision primarily based on Structural Layout (Weight: {1-alpha:.2f}).")
        
    # 4. Action Recommendation
    if prob > 0.8:
        explanations.append("Verdict: BLOCK - High-confidence phishing detection.")
    elif prob > 0.5:
        if uncertainty > 0.03:
            explanations.append("Verdict: WARN - Likely phishing, but with low model confidence. Manual review recommended.")
        else:
            explanations.append("Verdict: WARN - Suspicious activity detected.")
    else:
        explanations.append("Verdict: ALLOW - Normal URL patterns observed.")
        
    return explanations
