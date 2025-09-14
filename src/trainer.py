import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import logging

logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, model, optimizer, criterion,
                 train_loader, val_loader, save_path,
                 device=None, epochs=5, alpha=1.0,
                 patience=3, save_every=2, resume=False,start_epoch=1):
        """
        Args:
            model (nn.Module): BERT-based model.
            optimizer (torch.optim.Optimizer)
            criterion (nn.Module): Loss function.
            train_loader, val_loader (DataLoader)
            save_path (str): Directory to save checkpoints.
            device (str)
            epochs (int)
            alpha (float): Weighting for loss (if combining multiple losses)
            patience (int): Early stopping patience
            save_every (int): Save checkpoint every N epochs
            resume (bool): Resume training from last checkpoint if True
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.epochs = epochs
        self.alpha = alpha
        self.patience = patience
        self.save_every = save_every
        self.best_val_loss = float('inf')
        self.no_improve_counter = 0
        self.start_epoch = 1

        # Resume from checkpoint if available
        if resume:
            ckpt_file = os.path.join(self.save_path, "last_checkpoint.pt")
            if os.path.exists(ckpt_file):
                checkpoint = torch.load(ckpt_file, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.best_val_loss = checkpoint["best_val_loss"]
                self.start_epoch = checkpoint["epoch"] + 1
                logging.info(f"Resumed training from epoch {checkpoint['epoch']} (Val Loss={checkpoint['best_val_loss']:.4f})")
            else:
                logging.info("No checkpoint found. Starting from scratch.")

    def _validate(self):
        """Run validation and return loss and F1 score"""
        self.model.eval()
        losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                losses.append(loss.item())

                preds = outputs.argmax(dim=-1).cpu().numpy().flatten()
                labels_flat = labels.cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(labels_flat)

        avg_loss = np.mean(losses)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return avg_loss, f1

    def train(self):
        """Main training loop with validation and checkpointing"""
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            epoch_losses = []
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")

            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix({"train_loss": np.mean(epoch_losses)})

            val_loss, val_f1 = self._validate()
            logging.info(f"Epoch {epoch} => Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improve_counter = 0
                ckpt_path = os.path.join(self.save_path, "best_model.pt")
                torch.save(self.model.state_dict(), ckpt_path)
                logging.info(f"✅ Saved BEST model checkpoint at {ckpt_path}")
            else:
                self.no_improve_counter += 1
                if self.no_improve_counter >= self.patience:
                    logging.info(f"⏹ No improvement for {self.patience} epochs. Early stopping.")
                    break

            # Save checkpoint every N epochs
            if epoch % self.save_every == 0:
                ckpt_file = os.path.join(self.save_path, f"checkpoint_epoch{epoch}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "best_val_loss": self.best_val_loss,
                }, ckpt_file)
                logging.info(f"💾 Saved checkpoint at {ckpt_file}")

            # Always save latest checkpoint for resume
            last_ckpt_file = os.path.join(self.save_path, "last_checkpoint.pt")
            torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            }, last_ckpt_file)
