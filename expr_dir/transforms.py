import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(patch_size=512):
    """Трансформации для обучения. Изображение уже 512x512!"""
    return A.Compose(
        [
            # 🔧 НЕ НУЖНО Resize или Crop - изображение уже 512x512!
            # Только геометрические аугментации
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Легкие цветовые искажения
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            # Нормализация
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transforms(patch_size=512):
    """Трансформации для валидации."""
    return A.Compose(
        [
            # Без аугментаций, только нормализация
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
