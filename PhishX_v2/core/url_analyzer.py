import requests
import re
from urllib.parse import urlparse
import math
import string

class URLAnalyzer:
    def __init__(self):
        # List of suspicious TLDs (could be expanded)
        self.suspicious_tlds = {'.xyz', '.top', '.pw', '.biz', '.info', '.site', '.online', '.icu'}
        
        # List of known URL shorteners
        self.shorteners = {'bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'is.gd', 'buff.ly', 'ow.ly'}

    def expand_url(self, url):
        """
        Expand shortened URLs and follow redirects.
        """
        try:
            # Use a realistic User-Agent to avoid blocks
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            # Only head request to be efficient
            response = requests.head(url, headers=headers, allow_redirects=True, timeout=5)
            final_url = response.url
            return {
                "final_url": final_url,
                "is_redirected": final_url != url,
                "status_code": response.status_code
            }
        except Exception:
            # Fallback if request fails
            return {
                "final_url": url,
                "is_redirected": False,
                "status_code": None
            }

    def get_entropy(self, s):
        """
        Calculate Shannon entropy of a string.
        """
        if not s:
            return 0
        prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        return entropy

    def extract_features(self, url):
        """
        Extract structural features from URL for ML model.
        """
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        
        features = {
            "url_length": len(url),
            "dot_count": url.count('.'),
            "subdomain_count": domain.count('.') - 1 if domain.count('.') > 1 else 0,
            "is_ip": 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0,
            "has_special_chars": 1 if any(c in url for c in ['@', '?', '=', '&', '%']) else 0,
            "hyphen_count": url.count('-'),
            "entropy": self.get_entropy(url),
            "is_https": 1 if parsed.scheme == 'https' else 0,
            "suspicious_tld": 1 if any(url.endswith(tld) for tld in self.suspicious_tlds) else 0,
            "short_url": 1 if domain in self.shorteners else 0
        }
        
        # Convert dictionary to ordered feature vector for model input
        feature_vector = [
            features["url_length"],
            features["dot_count"],
            features["subdomain_count"],
            features["is_ip"],
            features["has_special_chars"],
            features["hyphen_count"],
            features["entropy"],
            features["is_https"],
            features["suspicious_tld"],
            features["short_url"]
        ]
        
        return features, feature_vector

    def process_url(self, url):
        """
        High-level wrapper for URL analysis.
        """
        expanded = self.expand_url(url)
        final_url = expanded["final_url"]
        features, feature_vector = self.extract_features(final_url)
        
        return {
            "original_url": url,
            "final_url": final_url,
            "is_shortened": expanded["is_redirected"],
            "features": features,
            "feature_vector": feature_vector
        }
