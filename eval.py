"""Evaluate an image classification model."""

import torch
from tqdm import tqdm
from utils import tqdm_bar


def evaluate(
    args,
    epoch: int,
    device: torch.device,
    model: torch.nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module
) -> tuple[torch.Tensor, float]:
    """
    Evaluate the model on the validation dataset.

    Args:
        args: Parsed command-line arguments.
        epoch: Current epoch number.
        device: Device to run the evaluation on.
        model: Trained model to evaluate.
        valid_loader: DataLoader for the validation set.
        criterion: Loss function.

    Returns:
        avg_loss (torch.Tensor): Average validation loss.
        avg_acc (float): Validation accuracy.
    """

    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for img, label in (pbar := tqdm(valid_loader, ncols=120)):
            img = img.to(device)
            label = label.to(device)

            pred = model(img)
            loss = criterion(pred, label)

            pred = torch.argmax(pred, dim=1)
            correct += (pred == label).sum().item()
            total_loss += loss

            tqdm_bar('Val', pbar, loss.detach().cpu(), epoch, args.epochs)

    avg_loss = total_loss / len(valid_loader)
    avg_acc = correct / len(valid_loader.dataset)

    return avg_loss, avg_acc
