from pathlib import Path

import torch
from dataset import ArchaeologyDataset
from torch.utils.data import DataLoader
from trainer import Trainer
from transforms import get_train_transforms, get_val_transforms

from config import Config, get_resnet34_unet_config
from models import create_model


def main():
    # Создаем конфигурацию
    config = get_resnet34_unet_config()

    # Настраиваем пути
    config.DATA_ROOT = Path("/home/kadafi/prog/archaeology-segmentation/data/raw")
    config.VAL_REGIONS = ["004_ДЕМИДОВКА", "080_Белая_Гора"]  # Укажите свои регионы для валидации
    config.EPOCHS = 50
    config.BATCH_SIZE = 2
    config.PATCH_SIZE = 512
    config.PATCHES_PER_IMAGE = 5  # Количество патчей с одного изображения

    # Создаем датасеты
    train_dataset = ArchaeologyDataset(
        root_dir=config.DATA_ROOT,
        transform=get_train_transforms(config.PATCH_SIZE),
        split="train",
        valid_regions=config.VAL_REGIONS,
        patch_size=config.PATCH_SIZE,
        patches_per_image=config.PATCHES_PER_IMAGE,
    )

    val_dataset = ArchaeologyDataset(
        root_dir=config.DATA_ROOT,
        transform=get_val_transforms(config.PATCH_SIZE),
        split="val",
        valid_regions=config.VAL_REGIONS,
        patch_size=config.PATCH_SIZE,
        patches_per_image=5,  # Меньше патчей для валидации
    )

    # Создаем загрузчики
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Создаем модель
    model = create_model(config)

    # Создаем тренер и запускаем обучение
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()


if __name__ == "__main__":
    main()
