# utils/trainer.py

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import wandb


# ─────────────────────────────────────────────
# Base Trainer
# ─────────────────────────────────────────────

class Trainer(ABC):
    def __init__(self, model: nn.Module, optimizer, criterion,
                 checkpoint_path: str, scheduler=None, use_wandb: bool = False):
        self.model           = model
        self.optimizer       = optimizer
        self.criterion       = criterion
        self.checkpoint_path = checkpoint_path
        self.scheduler       = scheduler
        self.use_wandb       = use_wandb

        # internal state
        self.best_val_metric  = float('inf')
        self.patience_counter = 0

    @abstractmethod
    def train_epoch(self, train_loader) -> tuple[float, float]:
        """Run one training epoch. Returns (loss, metric)."""
        ...

    @abstractmethod
    def val_epoch(self, val_loader) -> tuple[float, float]:
        """Run one validation epoch. Returns (loss, metric)."""
        ...

    @abstractmethod
    def evaluate(self, test_loader) -> tuple[float, float]:
        """Evaluate on test set. Returns (loss, metric)."""
        ...

    @abstractmethod
    def _wandb_metrics(self, train_loss: float, train_metric: float,
                       val_loss: float, val_metric: float) -> dict:
        """Return a dict of wandb metric keys → values for one epoch."""
        ...

    def fit(self, train_loader, val_loader,
            n_epochs: int, early_stop_patience: int = 15):

        for epoch in range(1, n_epochs + 1):
            train_loss, train_metric = self.train_epoch(train_loader)
            val_loss,   val_metric   = self.val_epoch(val_loader)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Console logging every 10 epochs
            if epoch == 1 or epoch % 10 == 0:
                self._log(epoch, train_loss, train_metric, val_loss, val_metric)

            # wandb logging every epoch
            if self.use_wandb:
                metrics = self._wandb_metrics(train_loss, train_metric, val_loss, val_metric)
                if self.scheduler:
                    metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                wandb.log(metrics, step=epoch)

            # Early stopping + checkpointing
            if val_loss < self.best_val_metric:
                self.best_val_metric  = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    self.model.load_state_dict(torch.load(self.checkpoint_path))
                    return

        # Load best model if early stopping didn't trigger
        self.model.load_state_dict(torch.load(self.checkpoint_path))

    @abstractmethod
    def _log(self, epoch, train_loss, train_metric,
             val_loss, val_metric):
        """Print epoch summary."""
        ...


# ─────────────────────────────────────────────
# Classification Trainer
# ─────────────────────────────────────────────

class ClassificationTrainer(Trainer):

    def train_epoch(self, train_loader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for Xb, yb in train_loader:
            pred = self.model(Xb)
            loss = self.criterion(pred, yb)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct    += (pred.argmax(1) == yb).sum().item()
            total      += yb.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def val_epoch(self, val_loader) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for Xb, yb in val_loader:
                out  = self.model(Xb)
                loss = self.criterion(out, yb)

                total_loss += loss.item()
                correct    += (out.argmax(1) == yb).sum().item()
                total      += yb.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def evaluate(self, test_loader) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for Xb, yb in test_loader:
                out  = self.model(Xb)
                loss = self.criterion(out, yb)

                total_loss += loss.item()
                correct    += (out.argmax(1) == yb).sum().item()
                total      += yb.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
        print(f"Test loss: {avg_loss:.4f} acc: {accuracy:.1f}%")

        if self.use_wandb:
            wandb.log({"test/loss": avg_loss, "test/accuracy": accuracy})

        return avg_loss, accuracy

    def _wandb_metrics(self, train_loss, train_metric, val_loss, val_metric) -> dict:
        return {
            "train/loss":     train_loss,
            "train/accuracy": train_metric,
            "val/loss":       val_loss,
            "val/accuracy":   val_metric,
        }

    def _log(self, epoch, train_loss, train_metric, val_loss, val_metric):
        print(
            f"Epoch {epoch:3d} | "
            f"Train loss: {train_loss:.4f}  acc: {train_metric:.1f}% | "
            f"Val loss: {val_loss:.4f}  acc: {val_metric:.1f}%"
        )


# ─────────────────────────────────────────────
# Regression Trainer
# ─────────────────────────────────────────────

class RegressionTrainer(Trainer):
    def __init__(self, model, optimizer, criterion,
                 checkpoint_path: str, std: float, scheduler=None, use_wandb: bool = False):
        super().__init__(model, optimizer, criterion, checkpoint_path, scheduler, use_wandb)
        self.std = std   # target std — for normalized RMSE metric

    def train_epoch(self, train_loader) -> tuple[float, float]:
        self.model.train()
        total_loss, total_nrmse = 0.0, 0.0

        for Xb, yb in train_loader:
            pred = self.model(Xb)
            loss = self.criterion(pred, yb.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_nrmse += loss.item() ** 0.5 / self.std

        avg_loss = total_loss / len(train_loader)
        avg_nrmse = total_nrmse / len(train_loader)
        return avg_loss, avg_nrmse

    def val_epoch(self, val_loader) -> tuple[float, float]:
        self.model.eval()
        total_loss, total_nrmse = 0.0, 0.0

        with torch.no_grad():
            for Xb, yb in val_loader:
                out  = self.model(Xb)
                loss = self.criterion(out, yb.unsqueeze(1))

                total_loss += loss.item()
                total_nrmse += loss.item() ** 0.5 / self.std

        avg_loss = total_loss / len(val_loader)
        avg_nrmse = total_nrmse / len(val_loader)
        return avg_loss, avg_nrmse

    def evaluate(self, test_loader) -> tuple[float, float]:
        self.model.eval()
        total_loss, total_rmse = 0.0, 0.0

        with torch.no_grad():
            for Xb, yb in test_loader:
                out  = self.model(Xb)
                loss = self.criterion(out, yb.unsqueeze(1))

                total_loss += loss.item()
                total_rmse += loss.item() ** 0.5 / self.std

        avg_loss = total_loss / len(test_loader)
        avg_nrmse = total_rmse / len(test_loader)
        print(f"Test loss: {avg_loss:.4f} NRMSE: {avg_nrmse:.4f}")

        if self.use_wandb:
            wandb.log({"test/loss": avg_loss, "test/nrmse": avg_nrmse})

        return avg_loss, avg_nrmse

    def _wandb_metrics(self, train_loss, train_metric, val_loss, val_metric) -> dict:
        return {
            "train/loss":  train_loss,
            "train/nrmse": train_metric,
            "val/loss":    val_loss,
            "val/nrmse":   val_metric,
        }

    def _log(self, epoch, train_loss, train_metric, val_loss, val_metric):
        print(
            f"Epoch {epoch:3d} | "
            f"Train loss: {train_loss:.4f}  NRMSE: {train_metric:.4f} | "
            f"Val loss: {val_loss:.4f}  NRMSE: {val_metric:.4f}"
        )