from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from losses import CombinedLoss
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = config.DEVICE
        self.model = self.model.to(self.device)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()

        self.scaler = GradScaler("cuda" if config.USE_AMP else "cpu")

        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []

        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.early_stop_counter = 0
        self.early_stop_best_loss = float("inf")

        self.use_wandb = config.USE_WANDB
        self._init_logging()

    def _create_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

    def _create_scheduler(self):
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.SCHEDULER_T0,
            T_mult=self.config.SCHEDULER_T_MULT,
            eta_min=self.config.MIN_LR,
        )

    def _create_criterion(self):
        # Вычисляем веса классов если не заданы
        if self.config.CLASS_WEIGHTS is None:
            class_weights = torch.ones(self.config.NUM_CLASSES)
            class_weights[0] = 0.1  # Уменьшаем вес фона
        else:
            class_weights = torch.tensor(self.config.CLASS_WEIGHTS)

        class_weights = class_weights.to(self.device)

        return CombinedLoss(class_weights=class_weights, dice_weight=self.config.DICE_WEIGHT)

    def _init_logging(self):
        if self.use_wandb:
            experiment_name = (
                f"{self.config.MODEL_NAME}_"
                f"{self.config.ENCODER}_"
                f"bs{self.config.BATCH_SIZE}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M')}"
            )

            wandb.init(
                project=self.config.WANDB_PROJECT,
                name=experiment_name,
                config=vars(self.config),
                dir=self.config.LOG_DIR,
            )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        print("num batches: ", num_batches)

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch}")

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device).long()

            self.optimizer.zero_grad()

            if self.config.USE_AMP:
                with autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                self.scaler.scale(loss).backward()

                if self.config.GRAD_CLIP > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()

                if self.config.GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

                self.optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "avg": f"{total_loss/(batch_idx+1):.4f}"}
            )

            if self.use_wandb and batch_idx % self.config.LOG_INTERVAL == 0:
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

        return total_loss / num_batches

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        # Накопители для confusion matrix
        iou_per_class = {i: [] for i in range(1, self.config.NUM_CLASSES)}

        pbar = tqdm(self.val_loader, desc=f"Val Epoch {self.current_epoch}")

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device).long()

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            # Считаем IoU для каждого класса
            for cls in range(1, self.config.NUM_CLASSES):
                pred_cls = predictions == cls
                true_cls = masks == cls

                intersection = (pred_cls & true_cls).sum().float()
                union = (pred_cls | true_cls).sum().float()

                if union > 0:
                    iou = (intersection / union).cpu().item()
                    iou_per_class[cls].append(iou)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Усредняем метрики
        avg_loss = total_loss / num_batches

        metrics = {}
        for cls in range(1, self.config.NUM_CLASSES):
            if iou_per_class[cls]:
                metrics[f"iou_class_{cls}"] = np.mean(iou_per_class[cls])
            else:
                metrics[f"iou_class_{cls}"] = 0.0

        metrics["mean_iou"] = np.mean([v for v in metrics.values()])

        return avg_loss, metrics

    def save_checkpoint(self, filename):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_metric": self.best_val_metric,
            "config": self.config,
        }

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def fit(self):
        print("\n" + "=" * 60)
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 60 + "\n")

        for epoch in range(1, self.config.EPOCHS + 1):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Scheduler step
            if not self.config.SCHEDULER_STEP_PER_BATCH:
                self.scheduler.step()

            # Validation
            if epoch % self.config.VAL_EVERY == 0:
                val_loss, val_metrics = self.validate_epoch()
                self.val_losses.append(val_loss)

                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val mIoU: {val_metrics['mean_iou']:.4f}")

                if self.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train/loss": train_loss,
                            "val/loss": val_loss,
                            "val/mean_iou": val_metrics["mean_iou"],
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        }
                    )

                # Save best model
                if val_metrics["mean_iou"] > self.best_val_metric:
                    self.best_val_metric = val_metrics["mean_iou"]
                    self.best_epoch = epoch
                    self.save_checkpoint("best_model.pth")
                    print(f"  ✅ New best model! mIoU: {self.best_val_metric:.4f}")

                # Early stopping
                if self.config.USE_EARLY_STOPPING:
                    if val_loss < self.early_stop_best_loss - self.config.EARLY_STOP_MIN_DELTA:
                        self.early_stop_best_loss = val_loss
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1

                    if self.early_stop_counter >= self.config.EARLY_STOP_PATIENCE:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break
            else:
                print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")

            # Periodic checkpoint
            if epoch % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

        self.save_checkpoint("final_model.pth")
        print("\n" + "=" * 60)
        print(
            f"Training completed! Best mIoU: {self.best_val_metric:.4f} at epoch {self.best_epoch}"
        )
        print("=" * 60)

        if self.use_wandb:
            wandb.finish()

        return self.model
