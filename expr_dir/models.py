import segmentation_models_pytorch as smp


def create_model(config):
    """Создает модель сегментации."""

    if config.MODEL_NAME == "Unet":
        model = smp.Unet(
            encoder_name=config.ENCODER,
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=config.NUM_CLASSES,
            activation=None,  # Без активации на выходе
        )
    elif config.MODEL_NAME == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=config.ENCODER,
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=config.NUM_CLASSES,
            activation=None,
        )
    elif config.MODEL_NAME == "FPN":
        model = smp.FPN(
            encoder_name=config.ENCODER,
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=config.NUM_CLASSES,
            activation=None,
        )
    else:
        raise ValueError(f"Unknown model name: {config.MODEL_NAME}")

    return model
