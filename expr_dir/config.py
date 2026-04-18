from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class Config:
    """Полная конфигурация для обучения модели сегментации археологических объектов."""

    # ========== Данные ==========
    DATA_ROOT: Path = Path("./data/raw")
    VAL_REGIONS: List[str] = field(default_factory=list)  # Список регионов для валидации

    # ========== Классы ==========
    NUM_CLASSES: int = 8  # 7 классов объектов + фон (0)
    CLASS_NAMES: List[str] = field(
        default_factory=lambda: [
            "kurgany_tselye",  # класс 1
            "kurgany_povrezhdennye",  # класс 2
            "fortifikatsii",  # класс 3
            "gorodishcha",  # класс 4
            "arkhitektury",  # класс 5
            "finds_points",  # класс 6
            "object_poly",  # класс 7
        ]
    )

    # Маппинг для определения класса по имени файла
    CLASS_MAPPING: dict = field(
        default_factory=lambda: {
            "курганы_целые": 1,
            "курганы_поврежденные": 2,
            "фортификации": 3,
            "городища": 4,
            "архитектуры": 5,
            "_FindsPoints": 6,
            "_ObjectPoly": 7,
        }
    )

    # ========== Патчи (нарезание больших изображений) ==========
    PATCH_SIZE: int = 512  # Размер патча (512x512 пикселей)
    PATCH_OVERLAP: int = 0  # Перекрытие патчей при инференсе (0 для обучения)
    PATCHES_PER_IMAGE: int = 5  # Сколько патчей нарезать с одного изображения за эпоху
    MIN_OBJECT_PIXELS: int = 100  # Минимальное количество пикселей объекта в патче (для фильтрации)

    # ========== Модель ==========
    MODEL_NAME: str = "Unet"  # Варианты: "Unet", "DeepLabV3Plus", "FPN", "Linknet", "PSPNet"
    ENCODER: str = (
        "resnet34"  # Варианты: "resnet18", "resnet34", "resnet50", "efficientnet-b0", etc.
    )
    ENCODER_WEIGHTS: str = "imagenet"  # "imagenet", "ssl", "swsl" или None
    IN_CHANNELS: int = 3  # RGB
    ACTIVATION: Optional[str] = None  # None для CrossEntropyLoss, "softmax" или "sigmoid"

    # ========== Обучение ==========
    EPOCHS: int = 100
    BATCH_SIZE: int = 2
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    GRAD_CLIP: float = 1.0  # Клиппирование градиентов (0 = отключено)

    # ========== Оптимизация памяти и скорости ==========
    USE_AMP: bool = True  # Mixed precision training (экономит память GPU)
    NUM_WORKERS: int = 1  # Количество потоков для DataLoader
    PIN_MEMORY: bool = True  # Ускоряет перенос данных на GPU
    PREFETCH_FACTOR: int = 0  # Предзагрузка батчей
    GRADIENT_ACCUMULATION_STEPS: int = (
        1  # Аккумуляция градиентов (>1 для имитации большего batch size)
    )

    # ========== Планировщик Learning Rate ==========
    SCHEDULER_NAME: str = (
        "CosineAnnealingWarmRestarts"  # "ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR"
    )
    SCHEDULER_T0: int = 20  # Период первого рестарта (для CosineAnnealingWarmRestarts)
    SCHEDULER_T_MULT: int = 2  # Множитель периода
    MIN_LR: float = 1e-6  # Минимальный learning rate
    SCHEDULER_STEP_PER_BATCH: bool = False  # True для OneCycleLR, False для остальных
    WARMUP_EPOCHS: int = 5  # Количество эпох для warmup (0 = отключено)
    WARMUP_START_LR: float = 1e-6  # Начальный LR для warmup

    # ========== Функция потерь ==========
    LOSS_NAME: str = "CombinedLoss"  # "CrossEntropy", "DiceLoss", "FocalLoss", "CombinedLoss"
    CLASS_WEIGHTS: Optional[List[float]] = None  # Если None - вычислятся автоматически
    DICE_WEIGHT: float = 0.5  # Вес Dice Loss в CombinedLoss
    FOCAL_GAMMA: float = 2.0  # Параметр gamma для Focal Loss
    FOCAL_ALPHA: Optional[List[float]] = None  # Параметр alpha для Focal Loss
    USE_CLASS_BALANCING: bool = True  # Автоматически балансировать веса классов
    BACKGROUND_WEIGHT_FACTOR: float = 0.1  # Множитель для уменьшения веса фона

    # ========== Валидация ==========
    VAL_EVERY: int = 1  # Валидация каждую N эпоху
    MONITOR_METRIC: str = "mean_iou"  # "mean_iou", "mean_dice", "val_loss"
    PRINT_PER_CLASS_METRICS: bool = True  # Выводить метрики по каждому классу

    # ========== Ранняя остановка ==========
    USE_EARLY_STOPPING: bool = True
    EARLY_STOP_PATIENCE: int = 15  # Количество эпох без улучшения
    EARLY_STOP_MIN_DELTA: float = 1e-4  # Минимальное улучшение
    EARLY_STOP_METRIC: str = "val_loss"  # "val_loss", "mean_iou"

    # ========== Сохранение чекпоинтов ==========
    CHECKPOINT_DIR: Path = Path("./checkpoints")
    SAVE_EVERY: int = 10  # Сохранять чекпоинт каждые N эпох
    SAVE_BEST_ONLY: bool = False  # Сохранять только лучшую модель
    SAVE_OPTIMIZER: bool = True  # Сохранять состояние оптимизатора для возобновления обучения
    RESUME_FROM: Optional[Path] = None  # Путь к чекпоинту для возобновления обучения

    # ========== Логирование ==========
    USE_WANDB: bool = False  # Использовать Weights & Biases
    WANDB_PROJECT: str = "archaeology_segmentation"
    WANDB_ENTITY: Optional[str] = None  # Имя пользователя/команды в wandb
    WANDB_LOG_MODEL: bool = True  # Логировать модель в wandb
    LOG_DIR: Path = Path("./logs")
    LOG_INTERVAL: int = 50  # Логировать в wandb каждые N батчей
    LOG_IMAGES: bool = False  # Логировать примеры сегментации
    LOG_IMAGES_EVERY: int = 5  # Логировать изображения каждые N эпох

    # ========== Аугментации ==========
    AUGMENTATION_INTENSITY: str = "medium"  # "light", "medium", "strong"
    USE_GEOMETRIC_AUGS: bool = True  # Flip, Rotate, etc.
    USE_COLOR_AUGS: bool = True  # Brightness, Contrast, etc.
    RANDOM_CROP: bool = True  # Использовать случайный кроп вместо центрального

    # Параметры аугментаций
    HORIZONTAL_FLIP_PROB: float = 0.5
    VERTICAL_FLIP_PROB: float = 0.5
    ROTATE_90_PROB: float = 0.5
    BRIGHTNESS_CONTRAST_PROB: float = 0.3
    BRIGHTNESS_LIMIT: float = 0.1
    CONTRAST_LIMIT: float = 0.1

    # ========== Устройство ==========
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_IDS: List[int] = field(default_factory=lambda: [0])  # Для multi-GPU обучения

    # ========== Воспроизводимость ==========
    SEED: int = 42
    DETERMINISTIC: bool = True  # Детерминированные операции (медленнее, но воспроизводимо)

    # ========== Отладка ==========
    DEBUG: bool = False  # Режим отладки (меньше данных, больше логов)
    OVERFIT_BATCHES: int = 0  # Количество батчей для оверфита (0 = отключено)
    PROFILE: bool = False  # Профилирование производительности

    def __post_init__(self):
        """Пост-инициализация для проверок и создания директорий."""
        # Создаем директории
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Преобразуем пути в Path если они строками
        if isinstance(self.DATA_ROOT, str):
            self.DATA_ROOT = Path(self.DATA_ROOT)
        if isinstance(self.RESUME_FROM, str):
            self.RESUME_FROM = Path(self.RESUME_FROM)

        # Проверка количества классов
        expected_classes = len(self.CLASS_NAMES) + 1  # +1 для фона
        if self.NUM_CLASSES != expected_classes:
            print(
                f"⚠️ Предупреждение: NUM_CLASSES={self.NUM_CLASSES}, но CLASS_NAMES содержит {len(self.CLASS_NAMES)} классов (+фон = {expected_classes})"
            )
            print(f"   Устанавливаю NUM_CLASSES = {expected_classes}")
            self.NUM_CLASSES = expected_classes

        # Проверка устройства
        if self.DEVICE == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA не доступна, переключаю на CPU")
            self.DEVICE = "cpu"
            self.USE_AMP = False

        # Настройка аугментаций в зависимости от интенсивности
        if self.AUGMENTATION_INTENSITY == "light":
            self.HORIZONTAL_FLIP_PROB = 0.3
            self.VERTICAL_FLIP_PROB = 0.3
            self.ROTATE_90_PROB = 0.3
            self.BRIGHTNESS_CONTRAST_PROB = 0.2
        elif self.AUGMENTATION_INTENSITY == "strong":
            self.HORIZONTAL_FLIP_PROB = 0.7
            self.VERTICAL_FLIP_PROB = 0.7
            self.ROTATE_90_PROB = 0.7
            self.BRIGHTNESS_CONTRAST_PROB = 0.5
            self.BRIGHTNESS_LIMIT = 0.2
            self.CONTRAST_LIMIT = 0.2

        # Установка seed для воспроизводимости
        if self.SEED is not None:
            import random

            import numpy as np

            random.seed(self.SEED)
            np.random.seed(self.SEED)
            torch.manual_seed(self.SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.SEED)
                torch.cuda.manual_seed_all(self.SEED)

            if self.DETERMINISTIC:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # Проверка валидационных регионов
        if not self.VAL_REGIONS:
            print("⚠️ VAL_REGIONS пуст! Все данные пойдут в train. Укажите регионы для валидации!")

    def get_class_name(self, class_id: int) -> str:
        """Возвращает название класса по его ID."""
        if class_id == 0:
            return "background"
        elif 1 <= class_id <= len(self.CLASS_NAMES):
            return self.CLASS_NAMES[class_id - 1]
        else:
            return f"unknown_class_{class_id}"

    def to_dict(self) -> dict:
        """Преобразует конфиг в словарь (для логирования)."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, (list, dict)):
                result[key] = value
            else:
                result[key] = value
        return result

    def print_summary(self):
        """Выводит сводку конфигурации."""
        print("\n" + "=" * 70)
        print(" " * 20 + "CONFIGURATION SUMMARY")
        print("=" * 70)

        print(f"\n📁 Data:")
        print(f"   Root: {self.DATA_ROOT}")
        print(f"   Val regions: {self.VAL_REGIONS if self.VAL_REGIONS else 'ALL (no split)'}")

        print(f"\n🏷️ Classes:")
        print(f"   Total: {self.NUM_CLASSES} (including background)")
        for i, name in enumerate(self.CLASS_NAMES, 1):
            print(f"   {i}: {name}")

        print(f"\n🔬 Model:")
        print(f"   Architecture: {self.MODEL_NAME}")
        print(f"   Encoder: {self.ENCODER}")
        print(f"   Weights: {self.ENCODER_WEIGHTS}")

        print(f"\n🎯 Training:")
        print(f"   Epochs: {self.EPOCHS}")
        print(f"   Batch size: {self.BATCH_SIZE}")
        print(f"   Patch size: {self.PATCH_SIZE}x{self.PATCH_SIZE}")
        print(f"   Learning rate: {self.LEARNING_RATE}")
        print(f"   Weight decay: {self.WEIGHT_DECAY}")
        print(f"   Device: {self.DEVICE}")
        print(f"   Mixed precision: {self.USE_AMP}")

        print(f"\n📊 Loss:")
        print(f"   Type: {self.LOSS_NAME}")
        print(f"   Dice weight: {self.DICE_WEIGHT}")
        print(f"   Class balancing: {self.USE_CLASS_BALANCING}")

        print(f"\n📈 Scheduler:")
        print(f"   Type: {self.SCHEDULER_NAME}")
        print(f"   Min LR: {self.MIN_LR}")

        print(f"\n💾 Checkpoints:")
        print(f"   Directory: {self.CHECKPOINT_DIR}")
        print(f"   Save every: {self.SAVE_EVERY} epochs")
        if self.RESUME_FROM:
            print(f"   Resume from: {self.RESUME_FROM}")

        print(f"\n🛑 Early stopping:")
        print(f"   Enabled: {self.USE_EARLY_STOPPING}")
        if self.USE_EARLY_STOPPING:
            print(f"   Patience: {self.EARLY_STOP_PATIENCE}")

        print(f"\n📝 Logging:")
        print(f"   WandB: {self.USE_WANDB}")
        if self.USE_WANDB:
            print(f"   Project: {self.WANDB_PROJECT}")

        print("\n" + "=" * 70)


# ========== Предустановленные конфигурации ==========


def get_resnet34_unet_config() -> Config:
    """Базовая конфигурация U-Net с ResNet34."""
    config = Config()
    config.MODEL_NAME = "Unet"
    config.ENCODER = "resnet34"
    config.BATCH_SIZE = 2
    config.LEARNING_RATE = 1e-4
    config.EPOCHS = 100
    return config


def get_efficientnet_unet_config() -> Config:
    """U-Net с EfficientNet-b3 (лучше качество, больше памяти)."""
    config = Config()
    config.MODEL_NAME = "Unet"
    config.ENCODER = "efficientnet-b3"
    config.BATCH_SIZE = 3
    config.LEARNING_RATE = 1e-4
    config.EPOCHS = 80
    config.PATCH_SIZE = 512
    return config


def get_deeplabv3plus_config() -> Config:
    """DeepLabV3+ с ResNet50."""
    config = Config()
    config.MODEL_NAME = "DeepLabV3Plus"
    config.ENCODER = "resnet50"
    config.BATCH_SIZE = 3
    config.LEARNING_RATE = 1e-4
    config.EPOCHS = 80
    return config


def get_fast_test_config() -> Config:
    """Конфигурация для быстрого тестирования."""
    config = Config()
    config.EPOCHS = 5
    config.BATCH_SIZE = 4
    config.PATCHES_PER_IMAGE = 5
    config.USE_WANDB = False
    config.USE_EARLY_STOPPING = False
    config.VAL_EVERY = 1
    config.SAVE_EVERY = 2
    config.DEBUG = True
    return config


def get_production_config() -> Config:
    """Конфигурация для production обучения."""
    config = Config()
    config.MODEL_NAME = "Unet"
    config.ENCODER = "efficientnet-b3"
    config.BATCH_SIZE = 8
    config.LEARNING_RATE = 3e-4
    config.EPOCHS = 150
    config.PATCH_SIZE = 512
    config.PATCHES_PER_IMAGE = 30
    config.USE_AMP = True
    config.USE_WANDB = True
    config.USE_EARLY_STOPPING = True
    config.EARLY_STOP_PATIENCE = 20
    config.AUGMENTATION_INTENSITY = "medium"
    config.WARMUP_EPOCHS = 5
    return config


# ========== Пример использования ==========
if __name__ == "__main__":
    # Создаем конфигурацию
    config = get_resnet34_unet_config()

    # Настраиваем под свои данные
    config.DATA_ROOT = Path("./data/regions")
    config.VAL_REGIONS = ["region_001", "region_042", "region_099"]
    config.CLASS_NAMES = [
        "kurgan_whole",
        "kurgan_damaged",
        "fortification",
        "settlement",
        "architecture",
        "find_point",
        "object_poly",
    ]

    # Выводим сводку
    config.print_summary()

    # Проверяем преобразование в словарь
    config_dict = config.to_dict()
    print(f"\nConfig as dict: {list(config_dict.keys())[:10]}...")
