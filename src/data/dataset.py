from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IMG_FOLD_NAME = "unet_dataset"
IMG_NAME = "image.npy"
MASK_NAME = "mask.npy"


class ArchaeologyDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train", valid_regions=None):
        """
        root_dir: Путь к папке, где лежат папки регионов (region_1, region_2...)
        transform: Аугментация из albumentations(библиотека)
        split: 'train' или 'val'
        valid_regions: список названий папок регионов для валидации (остальные в трейн)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []
        for dataset_dir in self.root_dir.glob("*/" + IMG_FOLD_NAME):
            img_path = dataset_dir / IMG_NAME
            mask_path = dataset_dir / MASK_NAME

            if img_path.exists() and mask_path.exists():
                # Определяем название региона (родительская папка)
                region_name = dataset_dir.parent.name
                # 2. Разделение на трейн/тест
                if valid_regions is not None:
                    if split == "train" and region_name not in valid_regions:
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                    elif split == "val" and region_name in valid_regions:
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                else:
                    # Если список валидации не указан, кладем всё в трейн (для первого теста)
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        print(f"Найдено {len(self.image_paths)} изображений для {split}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = np.load(img_path)  # Shape: (H, W, 3), dtype: uint8
        mask = np.load(mask_path)  # Shape: (H, W), dtype: uint8

        # 4. АУГМЕНТАЦИИ + трансформация
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Если трансформов нет, просто переводим в тензор
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask
