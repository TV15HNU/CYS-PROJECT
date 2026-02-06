## Dataset Usage

This project uses two phishing URL datasets:

1. **Kaggle Balanced URLs Dataset**

   * Used for initial BiLSTM training
   * Helps build character-level vocabulary

2. **StealthPhisher Dataset**

   * Used for benchmark training/testing split
   * Evaluated with Random Forest, BiLSTM, and Fusion models

All datasets are stored inside the `datasets/` folder.

# Hybrid Phishing URL Detection System

This project implements a hybrid phishing detection pipeline combining:

- Random Forest (feature-based ML)
- BiLSTM neural network
- Fusion ensemble model

## Results

Fusion Accuracy: 99.96%

The notebook demonstrates full benchmarking, evaluation, and visualization.

## Files

- PhishingProject_Report.ipynb — main notebook
- performance_metrics.csv — model comparison
- performance_chart.png — performance plot
- confusion_matrices.png — confusion matrices
