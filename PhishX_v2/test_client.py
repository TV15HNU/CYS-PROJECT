import requests
import os
import sys

def test_scan(image_path):
    url = "http://127.0.0.1:8000/scan"
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at {image_path}")
        return

    print(f"📤 Sending {image_path} to PhishX v2...")
    
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                print("✅ Scan Successful!")
                data = response.json()
                print("\n--- PhishX v2 Results ---")
                print(f"Status:      {data.get('status')}")
                print(f"URL:         {data.get('url')}")
                print(f"Final URL:   {data.get('final_url')}")
                print(f"Prediction:  {data.get('prediction')}")
                print(f"Risk Score:  {data.get('risk_score')}")
                print(f"Uncertainty: {data.get('uncertainty')}")
                print(f"Action:      {data.get('action')}")
                print(f"Alpha (Gate): {data.get('alpha')}")
                print("\n--- Features ---")
                for k, v in data.get('features', {}).items():
                    print(f"- {k}: {v}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to the PhishX server. Make sure it's running on port 8000.")

if __name__ == "__main__":
    # Default test image if none provided
    test_image = r"c:\Users\vishn\OneDrive\Desktop\cyber security\scan_20260211_105544_target.jpg"
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    test_scan(test_image)
