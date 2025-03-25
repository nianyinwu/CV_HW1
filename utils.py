"""Utility functions for training, evaluation and visualization."""

import os

from typing import List

import matplotlib.pyplot as plt
import torch.nn as nn



def tqdm_bar(mode: str, pbar, target: float = 0.0, cur_epoch: int = 0, epochs: int = 0) -> None:
    """
    Update the tqdm progress bar with custom format.

    Args:
        mode (str): Current mode ('Train', 'Val', 'Test').
        pbar: tqdm progress bar instance.
        target (float): Current loss value.
        cur_epoch (int): Current epoch.
        epochs (int): Total number of epochs.
    """
    if mode == 'Test':
        pbar.set_description(f"({mode})", refresh=False)
    else:
        pbar.set_description(f"({mode}) Epoch {cur_epoch}/{epochs}", refresh=False)
        pbar.set_postfix(loss=float(target), refresh=False)
    pbar.refresh()


def draw_figure(mode: str, path: str, epochs: int, train: List[float], valid: List[float]) -> None:
    """
    Plot and save training/validation loss or accuracy curve.

    Args:
        mode (str): 'loss' or 'accuracy'
        path (str): Path to save figure
        epochs (int): Total number of epochs
        train (List[float]): Training values per epoch
        valid (List[float]): Validation values per epoch
    """
    epoch_range = range(epochs)
    plt.style.use("ggplot")
    plt.figure()

    if mode == 'loss':
        plt.title("Loss")
        plt.ylabel("Loss")
        save_name = "Loss.jpg"
    elif mode == 'accuracy':
        plt.title("Accuracy")
        plt.ylabel("Accuracy")
        save_name = "Accuracy.jpg"
    else:
        raise ValueError("mode must be 'loss' or 'accuracy'")

    plt.plot(epoch_range, train, 'red', label='Training')
    plt.plot(epoch_range, valid, 'blue', label='Validation')
    plt.xlabel("Epoch")
    plt.legend()

    save_path = os.path.join(path, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"[DrawFigure] Saved figure to {save_path}")


def print_model_params(model: nn.Module) -> None:
    """
    Print the model architecture and total number of parameters.

    Args:
        model (nn.Module): PyTorch model
    """
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print("# Parameters:", total_params)
