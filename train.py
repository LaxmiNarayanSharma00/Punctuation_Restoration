import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer

from src.dataset import PunctuationDataset
from src.model import BertForPunctuation
from src.trainer import Trainer
from src.utils import set_all_seeds, write_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

CONFIG = {
    "seed": 42,
    "pretrained_bert": "bert-base-uncased",
    "dataset_path": "data/mental_health_dataset.csv",
    "save_path": "checkpoints",
    "batch_size": 16,
    "max_len": 64,
    "learning_rate": 3e-5,
    "epochs": 7,
    "alpha": 1.0,
    "patience": 3,
    "punc_to_class": {",": 1, ".": 2, "?": 3, "!": 4, ":": 5, ";": 6, "O": 0},
}

set_all_seeds(CONFIG["seed"])
logging.info(f"Seed set to {CONFIG['seed']}")

os.makedirs(CONFIG["save_path"], exist_ok=True)
write_yaml(CONFIG, os.path.join(CONFIG["save_path"], "config.yaml"))
logging.info(f"Configuration saved at {CONFIG['save_path']}/config.yaml")

# Tokenizer & Model
tokenizer = BertTokenizer.from_pretrained(CONFIG["pretrained_bert"])
model = BertForPunctuation(pretrained_model_name=CONFIG["pretrained_bert"],
                           num_punct_classes=len(CONFIG["punc_to_class"]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logging.info(f"Model loaded on device: {device}")

# Dataset & DataLoader
df = pd.read_csv(CONFIG["dataset_path"])
texts = df['Response'].dropna().tolist()

full_dataset = PunctuationDataset(
    texts=texts,
    tokenizer=tokenizer,
    class_to_punc=CONFIG["punc_to_class"],
    max_len=CONFIG["max_len"]
)

train_size = int(0.9 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
logging.info(f"Train size: {len(train_dataset)}, Validation size: {len(valid_dataset)}")

# Optimizer & loss
optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
criterion = torch.nn.CrossEntropyLoss()

# Trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=valid_loader,
    save_path=CONFIG["save_path"],
    device=device,
    epochs=CONFIG["epochs"],
    alpha=CONFIG["alpha"],
    patience=CONFIG["patience"]
)

logging.info("Starting training...")
trainer.train()
logging.info("Training finished!")
