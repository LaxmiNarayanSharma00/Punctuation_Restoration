import argparse
import os
import logging
import torch

from src.dataset import DataModule
from src.model import BertForPunctuation
from src.trainer import Trainer
from src.inference import PunctuationRestorer
from collections import Counter


def train(args):
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 🔹 Prepare data
    data_module = DataModule(
        csv_path=args.data_path,
        tokenizer_name="bert-base-uncased",
        max_len=args.max_len,
        batch_size=args.batch_size,
        val_size=0.1,
        test_size=0.1
    )
    train_loader, val_loader, _ = data_module.get_dataloaders()

    # Initialize model
    model = BertForPunctuation(
        pretrained_model_name="bert-base-uncased",
        num_punct_classes=len(data_module.class_to_punc)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 🔹 Compute class weights (from train_dataset labels)
    logging.basicConfig(level=logging.INFO)

    all_labels = []
    for batch in train_loader:   # loop once over train_loader
        labels = batch["labels"].view(-1).tolist()
        all_labels.extend(labels)

    label_counts = Counter(all_labels)
    num_classes = len(data_module.class_to_punc)
    total = sum(label_counts.values())
    class_weights = []
    for i in range(num_classes):
        freq = label_counts.get(i, 1)
        weight = total / (num_classes * freq)
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    logging.info(f"Class Weights: {class_weights}")

    # Optimizer & Weighted Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    start_epoch = 0

    # 🔹 Resume from checkpoint if provided
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logging.info(f"Resumed training from {args.resume_from}, starting at epoch {start_epoch}")

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=args.checkpoint_dir,
        device=device,
        epochs=args.epochs,
        save_every=args.save_every,
        start_epoch=start_epoch
    )
    trainer.train()


def inference(args):
    restorer = PunctuationRestorer(model_path=args.model_path)
    restored_text = restorer.restore_punctuation(args.text)
    print("Input Text: ", args.text)
    print("Restored Text: ", restored_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Punctuation Restoration System")

    subparsers = parser.add_subparsers(dest="command")

    # Training parser
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset")
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=7)
    train_parser.add_argument("--lr", type=float, default=3e-5)
    train_parser.add_argument("--resume_from", type=str, default=None,
                              help="Path to checkpoint to resume training from")
    train_parser.add_argument("--save_every", type=int, default=2,
                              help="Save checkpoint every N epochs")
    train_parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")

    # Inference parser
    infer_parser = subparsers.add_parser("inference")
    infer_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    infer_parser.add_argument("--text", type=str, required=True, help="Unpunctuated input text")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "inference":
        inference(args)
    else:
        print("Please specify a command: train or inference")
