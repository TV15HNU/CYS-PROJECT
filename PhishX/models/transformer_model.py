import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer

class URLTransformer(nn.Module):
    def __init__(self, num_classes=1):
        super(URLTransformer, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token output (index 0)
        pooled_output = output.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def get_tokenizer():
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
