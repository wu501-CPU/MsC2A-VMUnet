import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    def binary_cross_entropy_loss(y_true, y_pred):
        """
        Calculate binary cross-entropy loss
        y_true: True labels, numpy array
        y_pred: Predicted probabilities, numpy array
        """
        epsilon = 1e-15  # To prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predicted values between epsilon and 1-epsilon
        loss = - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def dice_loss(y_true, y_pred):
        """
        Calculate Dice loss
        y_true: True labels, numpy array (must be binary, i.e., 0 and 1)
        y_pred: Predicted probabilities, numpy array (values between 0 and 1)
        """
        y_pred_binary = np.round(y_pred)  # Convert predicted probabilities to binary labels
        intersection = np.sum(y_true * y_pred_binary)
        union = np.sum(y_true) + np.sum(y_pred_binary)
        loss = 1 - (2 * intersection) / union
        return loss

    def total_loss(y_true, y_pred, lambda_1=1.0, lambda_2=1.0):
        """
        Calculate total loss
        y_true: True labels, numpy array
        y_pred: Predicted probabilities, numpy array
        lambda_1: Weight for binary cross-entropy loss
        lambda_2: Weight for Dice loss
        """
        bce = binary_cross_entropy_loss(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)
        loss = lambda_1 * bce + lambda_2 * dice
        return loss
