import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import pandas as pd


class PunctuationDataset(Dataset):
    """PyTorch Dataset for punctuation restoration."""

    def __init__(self, texts, tokenizer, class_to_punc, max_len=128):
        """
        Args:
            texts (list[str]): List of sentences.
            tokenizer (transformers.PreTrainedTokenizer)
            class_to_punc (dict): {punct: class_id} mapping
            max_len (int): Maximum token length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.class_to_punc = class_to_punc
        self.punc_to_class = class_to_punc  # keep same mapping
        self.max_len = max_len

        self.data = [self._preprocess_text(t) for t in self.texts]

    def _preprocess_text(self, text):
        tokens, labels = [], []
        words = text.strip().split()

        for word in words:
            punc = ''
            if word[-1] in self.punc_to_class:
                punc = word[-1]
                word = word[:-1]

            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)

            for i in range(len(word_tokens)):
                labels.append(self.punc_to_class.get(punc, 0) if i == len(word_tokens) - 1 else self.punc_to_class['O'])

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # truncate & pad
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]

        attention_mask = [1] * len(input_ids)
        pad_len = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels += [0] * pad_len
        attention_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModule:
    """Handles dataset splitting and DataLoader creation."""

    def __init__(self, csv_path, tokenizer_name='bert-base-uncased', max_len=128,
                 batch_size=32, test_size=0.1, val_size=0.1):
        self.csv_path = csv_path
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size

        # Define punctuation classes
        self.class_to_punc = {",": 1, ".": 2, "?": 3, "!": 4, ":": 5, ";": 6, "O": 0}

        self._prepare_data()

    def _prepare_data(self):
        df = pd.read_csv(self.csv_path).dropna(subset=['Response'])
        texts = df['Response'].tolist()

        train_texts, temp_texts = train_test_split(
            texts, test_size=self.val_size + self.test_size, random_state=42
        )
        val_texts, test_texts = train_test_split(
            temp_texts, test_size=self.test_size / (self.val_size + self.test_size), random_state=42
        )

        self.train_dataset = PunctuationDataset(train_texts, self.tokenizer, self.class_to_punc, max_len=self.max_len)
        self.val_dataset = PunctuationDataset(val_texts, self.tokenizer, self.class_to_punc, max_len=self.max_len)
        self.test_dataset = PunctuationDataset(test_texts, self.tokenizer, self.class_to_punc, max_len=self.max_len)

    def get_dataloaders(self):
        """Returns train_loader, val_loader, test_loader compatible with Trainer"""
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
