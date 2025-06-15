import math
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import fsspec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from Resnet_18 import ResNet2d_18


@dataclass
class TrainerConfig:
    batch_size: int = None
    num_workers: int = None
    save_every: int = None
    epochs: int = None
    use_amp: bool = None
    grad_norm: float = None
    resume: bool = None
    snapshot_path: str = None


@dataclass
class OptimizerConfig:
    init_lr: float = None
    betas: tuple = None
    min_lr: float = None
    decay_lr: bool = None
    lr_decay_iters: float = None
    warmup_steps: int = None


@dataclass
class Snapshot:
    model_state: "OrderedDict[str,torch.Tensor]"
    optimizer_state: Dict[str, Any]
    finished_epoch: int


# save file
def save_func(data, name):
    with open(name, "a") as f:
        f.write(f"{data}\n")


# #this measure Precison, ReCall,F1-score
# def eval_matrics(predictions,targets):


class Trainer(nn.Module):
    def __init__(
        self,
        model: ResNet2d_18,
        trainer_config: TrainerConfig,
        optimizer_config: OptimizerConfig,
        optimizer,
        train_data,
        val_data,
    ):
        super().__init__()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.model = model.to(self.local_rank)
        self.optimizer_config = optimizer_config
        self.optimizer = optimizer
        self.trainer_config = trainer_config
        self.train_loader = self._prepare_dataloader(train_data, train=True)
        self.val_loader = self._prepare_dataloader(val_data, train=False)
        self.save_every = self.trainer_config.save_every
        self.epochs_run = 0
        if self.trainer_config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        if self.trainer_config.resume:
            self._load_snapshot()

    def _prepare_dataloader(self, dataset: Dataset, train=True):
        return DataLoader(
            dataset,
            batch_size=self.trainer_config.batch_size,
            shuffle=train,
            num_workers=self.trainer_config.num_workers,
            pin_memory=True,
        )

    def _load_snapshot(self):
        try:
            snapshot = fsspec(self.trainer_config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            print("snapshot not found, train from scratch")
            return

        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        print(f"Resume training at {self.epochs_run}")

    # schedual learning rate
    def get_lr(self, it):
        # warnup phase
        if it < self.optimizer_config.warmup_steps:
            return (
                it / self.optimizer_config.warmup_steps
            ) * self.optimizer_config.init_lr
        # after decaying
        elif it > self.optimizer_config.lr_decay_iters:
            return self.optimizer_config.min_lr
        diff_lr = self.optimizer_config.init_lr - self.optimizer_config.min_lr
        coeff_lr = (it - self.optimizer_config.warmup_steps) / (
            self.optimizer_config.lr_decay_iters - self.optimizer_config.warmup_steps
        )
        out_lr = self.optimizer_config.min_lr + 0.5 * diff_lr * (
            1 + math.cos(math.pi * coeff_lr)
        )
        return out_lr

    def _run_batch(self, source, targets, train: bool = True) -> float:
        with torch.set_grad_enabled(train), torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(self.trainer_config.use_amp),
        ):
            predicts = self.model(source)
            loss = F.cross_entropy(predicts, targets)
            probs = torch.softmax(predicts, dim=-1)
            acc = (torch.argmax(probs, dim=-1) == targets).sum().item()

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.trainer_config.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.trainer_config.grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.trainer_config.grad_norm
                )
                self.optimizer.step()
        return loss.item(), acc

    def _save_snapshot(self, epoch):
        model = self.model
        snapshot = Snapshot(
            model_state=model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
        )
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.trainer_config.snapshot_path)
        print(f"save snapshot at {epoch}")

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool, it: int):

        total_loss = 0.0
        total_it = 0
        # accuracy
        total_acc = 0
        step_type = "Train" if train else "Eval"
        for iter, (source, target) in enumerate(dataloader):
            lr = (
                self.get_lr(it)
                if self.optimizer_config.decay_lr
                else self.optimizer_config.init_lr
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            source, target = source.to(self.local_rank), target.to(self.local_rank)
            loss, acc = self._run_batch(source, target, train)
            total_loss += loss
            total_acc += acc
            total_it += 1
            it += 1
            if iter % 20 == 0:
                print(
                    f"Epoch {epoch} | Iter {iter} | {step_type} | Iter Loss {loss} | Iter accuracy {acc}"
                )
        avg_loss = total_loss / total_it
        avg_acc = total_acc / total_it
        return avg_loss, it, avg_acc, lr

    def train(self):
        it = 0
        eval_best = 0.0
        for epoch in range(self.trainer_config.epochs):
            epoch += 1
            avg_loss, it, avg_acc, lr = self._run_epoch(
                epoch, self.train_loader, train=True, it=it
            )
            print(f"Epoch{epoch} | loss per epoch {avg_loss} | acc per epoch {avg_acc}")
            save_func(avg_loss, "train_loss.txt")
            save_func(avg_acc, "train_acc.txt")
            save_func(lr, "lr_schedule.txt")
            if epoch % self.trainer_config.save_every == 0:
                eval_loss, it, eval_acc, _ = self._run_epoch(
                    epoch, self.val_loader, train=False, it=it
                )
                save_func(eval_loss, "eval_loss.txt")
                save_func(eval_acc, "eval_acc.txt")
                print(f"eval loss {eval_loss} | eval_acc {eval_acc}")
                if eval_best < eval_acc:
                    eval_best = eval_acc
                    self._save_snapshot(epoch)
