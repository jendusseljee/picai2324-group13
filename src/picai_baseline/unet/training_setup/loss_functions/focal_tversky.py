import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss function for binary segmentation."""

    def __init__(self, alpha=1, beta=0.3, gamma=2, epsilon=1e-6, reduction="sum"):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = epsilon

    def forward(self, inputs, targets):
        # Flatten predictions and targets
        inputs = torch.sigmoid(inputs)  # Ensure predictions are in [0, 1]
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute TP, FP, FN (true positives, false positives, false negatives)
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        # Tversky index
        tversky = (TP + self.epsilon) / (TP + self.alpha * FP + self.beta * FN + self.epsilon)

        # Focal Tversky loss
        loss = (1 - tversky) ** self.gamma

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

