import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """Комбинированный loss: CrossEntropy + Dice."""

    def __init__(self, class_weights=None, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    def dice_loss(self, pred, target, smooth=1e-7):
        """
        Soft Dice Loss для многоклассовой сегментации.
        """
        # pred: [B, C, H, W]
        # target: [B, H, W]

        pred_softmax = F.softmax(pred, dim=1)

        # One-hot encoding для target
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Вычисляем Dice для каждого класса
        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)

        # Усредняем по всем классам кроме фона (класс 0)
        dice_score = dice[:, 1:].mean()

        return 1 - dice_score

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return ce + self.dice_weight * dice
