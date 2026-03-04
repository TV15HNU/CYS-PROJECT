import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

# ---------------------------------------------------------
# Module 6A: Semantic Engine (Transformer)
# ---------------------------------------------------------
class SemanticEngine(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', dropout_rate=0.1):
        super(SemanticEngine, self).__init__()
        self.transformer = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        # We want to keep dropout active during inference for MC Dropout
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits)

# ---------------------------------------------------------
# Module 6B: Structural Engine (Character CNN)
# ---------------------------------------------------------
class CharacterCNN(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, dropout_rate=0.1):
        super(CharacterCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(2)  # (batch, 64)
        x = self.dropout(x)
        logits = self.fc(x)
        return torch.sigmoid(logits)

# ---------------------------------------------------------
# Module 7: Adaptive Gating Network
# ---------------------------------------------------------
class AdaptiveGatingNetwork(nn.Module):
    def __init__(self, input_dim=5):
        """
        Input: [mean_trans, var_trans, mean_cnn, var_cnn, metadata_score]
        Output: α (weight for transformer)
        """
        super(AdaptiveGatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        alpha = torch.sigmoid(self.fc2(x))
        return alpha

# ---------------------------------------------------------
# Module 6C & 8: PhishX Multi-Modal Wrapper
# ---------------------------------------------------------
class PhishXCore:
    def __init__(self, model_dir='models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Models
        self.semantic_engine = SemanticEngine().to(self.device)
        self.structural_engine = CharacterCNN().to(self.device)
        self.gating_network = AdaptiveGatingNetwork(input_dim=4).to(self.device)
        
        # Tokenizers / Preprocessing
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.char_map = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;%=")}

    def mc_dropout_predict(self, model, input_data, passes=10):
        """
        Perform Multiple MC Dropout passes to estimate uncertainty.
        """
        model.train()  # Keep dropout active
        predictions = []
        with torch.no_grad():
            for _ in range(passes):
                if isinstance(input_data, tuple):
                    pred = model(*input_data)
                else:
                    pred = model(input_data)
                predictions.append(pred.cpu().numpy())
        
        preds = torch.tensor(predictions)
        mean = torch.mean(preds).item()
        variance = torch.var(preds).item()
        return mean, variance

    def predict(self, url):
        """
        Full prediction pipeline.
        """
        # Preprocess for Transformer
        inputs = self.tokenizer(url, return_tensors='pt', padding=True, truncation=True, max_length=256).to(self.device)
        
        # Preprocess for CNN
        char_indices = torch.tensor([[self.char_map.get(c, 0) for c in url.lower()[:256]]]).to(self.device)
        
        # MC Dropout Predictions
        mu_trans, var_trans = self.mc_dropout_predict(self.semantic_engine, (inputs['input_ids'], inputs['attention_mask']))
        mu_cnn, var_cnn = self.mc_dropout_predict(self.structural_engine, char_indices)
        
        # Gating Network
        gate_input = torch.tensor([[mu_trans, var_trans, mu_cnn, var_cnn]]).float().to(self.device)
        alpha = self.gating_network(gate_input).item()
        
        # Risk Fusion
        final_risk = alpha * mu_trans + (1 - alpha) * mu_cnn
        total_uncertainty = (var_trans + var_cnn) / 2 # simplified
        
        return final_risk, total_uncertainty, alpha
