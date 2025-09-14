import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertForPunctuation(nn.Module):
    """
    BERT-based model for punctuation restoration.
    Predicts punctuation for each token.
    """
    def __init__(self, pretrained_model_name='bert-base-uncased', num_punct_classes=7, dropout=0.3):
        """
        Args:
            pretrained_model_name (str): HuggingFace BERT model name.
            num_punct_classes (int): Number of punctuation classes including 'O' (no punctuation).
            dropout (float): Dropout rate before classification layer.
        """
        super(BertForPunctuation, self).__init__()

        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout)

        # Token-level classification head
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_punct_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids (torch.LongTensor): Token IDs [batch_size, seq_len]
            attention_mask (torch.LongTensor): Mask for padded tokens
            token_type_ids (torch.LongTensor): Segment IDs if needed

        Returns:
            logits (torch.FloatTensor): [batch_size, seq_len, num_punct_classes]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)    # [batch_size, seq_len, num_punct_classes]
        return logits
