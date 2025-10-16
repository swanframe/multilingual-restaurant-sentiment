# src/models/modeling.py
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BertForSentiment(nn.Module):
    def __init__(self, pretrained_name: str, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.pretrained_name = pretrained_name  # Store for saving
        self.config = AutoConfig.from_pretrained(pretrained_name, num_labels=num_labels)
        self.bert = AutoModel.from_pretrained(pretrained_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = outputs.last_hidden_state[:, 0]  # CLS
        x = self.dropout(pooled)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}