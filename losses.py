import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.flatten(1)
        target = target.flatten(1)
        intersection = (pred * target).sum(1)
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(1) + target.sum(1) + self.smooth
        )
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + \
               self.dice_weight * self.dice(pred, target)


class StructureLoss(nn.Module):

    def forward(self, pred, target):
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
        )

        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred_sig = torch.sigmoid(pred)
        inter = ((pred_sig * target) * weit).sum(dim=(2, 3))
        union = ((pred_sig + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()
