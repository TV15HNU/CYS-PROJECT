# PhishX – QR Phishing Detection

## Overview

PhishX is a machine learning-based cybersecurity project designed to detect phishing URLs hidden inside QR codes, commonly known as **Quishing (QR Phishing)**. Since users cannot directly verify the destination of a QR code before scanning, attackers often exploit this to redirect users to fake login pages, malicious websites, or credential theft portals.

PhishX helps identify such malicious QR codes by extracting the embedded URL, analyzing its structure and behavior, and classifying it as safe or phishing using deep learning techniques.

---

## Problem Statement

QR code phishing attacks are increasing rapidly because users trust QR codes without being able to inspect the hidden destination link.

Traditional phishing detection systems often fail because:

* QR content is hidden until scanned
* Malicious links are obfuscated to bypass filters
* Static blacklists cannot detect newly created phishing URLs
* Adversarial attacks evade rule-based detection systems

PhishX solves this by combining QR code analysis with phishing URL detection for stronger and more reliable security.

---

## Key Features

### QR Code URL Extraction

Detects and decodes URLs embedded inside QR codes

### URL Feature Analysis

Analyzes URL structure, domain patterns, suspicious keywords, redirects, and lexical features

### Phishing Detection Model

Uses machine learning and deep learning models to classify URLs as benign or malicious

### Adversarial Attack Handling

Improves robustness against manipulated phishing URLs designed to bypass detection systems

### Dataset Support

Trained using balanced datasets of malicious and benign QR codes and phishing URLs

---

## Tech Stack

### Languages

* Python
* Jupyter Notebook

### Libraries & Tools

* TensorFlow / PyTorch
* Scikit-learn
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Requests
* BeautifulSoup
* QR Decoding Libraries

---

## Project Structure

```bash id="k2m4n8"
PhishX/
│
├── PhishX/                         # Core phishing detection model
├── PhishX_v2/                      # Improved robust detection version
├── PhishingProject/                # Dataset preprocessing and experiments
├── PhishingProjectFinal/           # Final implementation
├── QuishingDataset/                # QR phishing datasets
├── QR/QR codes/                    # QR code samples
├── StealthPhisher/                 # Adversarial phishing experiments
├── urldata/                        # URL datasets
├── balancedurls's/                 # Balanced benign/malicious URLs
├── malicious6L/                    # Malicious URL samples
├── Dataset of 1000 Images...       # QR image dataset
│
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Installation

### Clone Repository

```bash id="p9w1x4"
git clone https://github.com/srikanth0766/PhishX-Adversarially-Aware-QR-Phishing-Detection.git
cd PhishX-Adversarially-Aware-QR-Phishing-Detection
```

### Create Virtual Environment

```bash id="r5v2z7"
python -m venv venv
```

### Activate Virtual Environment

#### macOS / Linux

```bash id="t8c6m1"
source venv/bin/activate
```

#### Windows

```bash id="u3f9k2"
venv\Scripts\activate
```

### Install Dependencies

```bash id="w7n4b5"
pip install -r requirements.txt
```

---

## How It Works

### Step 1: Input QR Code

Upload a QR code image for analysis

### Step 2: Decode QR Content

Extract the hidden URL from the QR code

### Step 3: Feature Extraction

Generate lexical, structural, and behavioral features from the URL

### Step 4: Classification

Use ML/DL models to classify the URL as:

* Safe
* Suspicious
* Phishing

### Step 5: Threat Detection

Flag malicious QR codes before users interact with harmful websites

---

## Results

PhishX improves phishing detection by:

* Detecting malicious URLs hidden inside QR codes
* Reducing false positives compared to traditional URL filters
* Identifying suspicious links before user interaction
* Improving robustness against newly generated phishing attacks

This helps strengthen protection against credential theft, fake payment pages, and login phishing attacks delivered through QR codes.

---

## Future Improvements

* Real-time mobile QR scanner integration
* Browser extension for phishing prevention
* Live URL reputation checking
* Cloud deployment for API-based detection
* Advanced transformer-based phishing detection models

---

## License

This project is developed for academic and cybersecurity research purposes.
