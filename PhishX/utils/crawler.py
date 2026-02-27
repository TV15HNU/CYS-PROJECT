import requests
from bs4 import BeautifulSoup

def crawl_url(url):
    try:
        # Add a realistic User-Agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code != 200:
            return {"status": "failed", "error": f"HTTP {response.status_code}"}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string.strip() if soup.title else "No Title"
        
        meta_desc = ""
        description_tag = soup.find('meta', attrs={'name': 'description'}) or \
                          soup.find('meta', attrs={'property': 'og:description'})
        if description_tag:
            meta_desc = description_tag.get('content', '').strip()
            
        return {
            "status": "success",
            "title": title,
            "meta_description": meta_desc[:200]
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}
