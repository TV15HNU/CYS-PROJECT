import random
import re
from urllib.parse import urlparse, urlunparse

class AdversarialURLGenerator:
    """
    Modular Adversarial URL Generator for Cybersecurity Robustness Testing.
    """
    def __init__(self):
        # Homoglyphs: Latin to visually similar Cyrillic/Unicode characters
        self.homoglyphs = {
            'a': 'а', 'e': 'е', 'i': 'і', 'o': 'о', 'p': 'р', 
            'c': 'с', 'y': 'у', 'x': 'х', 'j': 'ј'
        }
        
        self.suspicious_prefixes = ['secure-', 'login-', 'update-', 'verify-', 'official-', 'signin-']
        self.malicious_tlds = ['.tk', '.xyz', '.ru', '.pw', '.top', '.ga', '.ml']

    def homoglyph_attack(self, url):
        """Replace latin characters with lookalikes."""
        new_url = ""
        for char in url:
            if char in self.homoglyphs and random.random() < 0.3:
                new_url += self.homoglyphs[char]
            else:
                new_url += char
        return new_url

    def typosquatting_attack(self, url):
        """Character swap, insertion, or deletion."""
        parsed = urlparse(url)
        domain = parsed.netloc
        if not domain: return url
        
        attack_type = random.choice(['swap', 'insert', 'delete'])
        if len(domain) < 4: return url
        
        pos = random.randint(0, len(domain) - 2)
        domain_list = list(domain)
        
        if attack_type == 'swap':
            domain_list[pos], domain_list[pos+1] = domain_list[pos+1], domain_list[pos]
        elif attack_type == 'insert':
            domain_list.insert(pos, random.choice('abcdefghijklmnopqrstuvwxyz'))
        elif attack_type == 'delete':
            domain_list.pop(pos)
            
        new_domain = "".join(domain_list)
        return url.replace(domain, new_domain)

    def subdomain_flooding_attack(self, url):
        """Insert deceptive subdomains."""
        parsed = urlparse(url)
        if not parsed.netloc: return url
        deceptive = random.choice(['secure-login', 'account-update', 'verification-portal', 'bank-security'])
        new_netloc = f"{deceptive}.{parsed.netloc}"
        return url.replace(parsed.netloc, new_netloc)

    def tld_squatting_attack(self, url):
        """Replace common TLDs with malicious ones."""
        for tld in ['.com', '.org', '.net', '.edu']:
            if url.endswith(tld) or f"{tld}/" in url:
                return url.replace(tld, random.choice(self.malicious_tlds))
        return url

    def prefix_injection_attack(self, url):
        """Add suspicious prefixes to the domain."""
        parsed = urlparse(url)
        if not parsed.netloc: return url
        prefix = random.choice(self.suspicious_prefixes)
        new_netloc = prefix + parsed.netloc
        return url.replace(parsed.netloc, new_netloc)

    def generate_all(self, url):
        """Apply a random attack."""
        attack = random.choice([
            self.homoglyph_attack, 
            self.typosquatting_attack, 
            self.subdomain_flooding_attack, 
            self.tld_squatting_attack,
            self.prefix_injection_attack
        ])
        return attack(url)
