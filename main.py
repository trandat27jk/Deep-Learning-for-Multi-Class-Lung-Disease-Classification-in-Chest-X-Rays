import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import random_split

from data_process import test_dataset, train_dataset, val_dataset
from Resnet_18 import ResNet2d_18, VisionConfig, create_optimizers
from trainer import OptimizerConfig, Trainer, TrainerConfig


def get_train_objs(vision_config: VisionConfig, optimizer_config):
    model = ResNet2d_18(vision_config)
    optimizer = create_optimizers(optimizer_config)
    return model, optimizer


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    vision_config = VisionConfig(**cfg["vision_config"])
    optimizer_config = OptimizerConfig(**cfg["optim_config"])
    trainer_config = TrainerConfig(**cfg["trainer_config"])
    model, optimizer = get_train_objs(vision_config, optimizer_config)
    trainer = Trainer(
        model, trainer_config, optimizer_config, optimizer, train_dataset, val_dataset
    )
    trainer.train()


if __name__ == "__main__":
    main()
