"""Training an image classification model."""

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from model import get_model
from eval import evaluate
from dataloader import dataloader
from utils import tqdm_bar, draw_figure


# Enable fast training
cudnn.benchmark = True


def get_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        '--device',
        type=str,
        choices=[
            "cuda",
            "cpu"],
        default="cuda"
    )
    parser.add_argument(
        '--data_path',
        '-d',
        type=str,
        default='./data',
        help='path of the input data'
    )
    parser.add_argument(
        '--save_path',
        '-s',
        type=str,
        default='./saved_model',
        help='path to save the training model'
    )
    parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        default=50,
        help='number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=16,
        help='batch size'
    )
    parser.add_argument(
        '--learning_rate',
        '-lr',
        type=float,
        default=1e-4,
        help='learning rate'
    )


    return parser.parse_args()


def train(
    args: argparse.Namespace,
    cur_epoch: int,
    train_device: torch.device,
    train_model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[torch.Tensor, float]:
    """
    Train the model for one epoch.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        cur_epoch (int): Current training epoch.
        train_device (torch.device): Device to train on (CPU or GPU).
        train_model (nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for training.

    Returns:
        Tuple[torch.Tensor, float]: The average training loss and accuracy.
    """

    total_loss = 0
    correct = 0
    for img, label in (pbar := tqdm(data_loader, ncols=120)):

        # Set image and label to the same device as the model
        img = img.to(device=train_device)
        label = label.to(device=train_device)

        # clear gradient
        optimizer.zero_grad()

        # input the data to model
        pred = train_model(img)

        # Loss Function
        loss = criterion(pred, label)

        # Calculate Total Loss
        total_loss += loss

        # Calculate Top1 Accuracy
        pred = torch.argmax(pred, dim=1)
        correct += (pred == label).sum().item()

        # Backward
        loss.backward()

        # Update all parameters
        optimizer.step()

        # Clear Gradient
        optimizer.zero_grad()

        tqdm_bar('Train', pbar, loss.detach().cpu(), cur_epoch, args.epochs)


    # Calculate Average Loss of current epoch
    avg_loss = total_loss / len(data_loader)
    avg_acc = correct / len(data_loader.dataset)
    return avg_loss, avg_acc


if __name__ == "__main__":
    opt = get_args()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    # Ensure the save path exist
    os.makedirs(opt.save_path, exist_ok=True)

    model = get_model().to(device)

    # Setting the loss function, optimizer and scheduler
    loss_func = nn.CrossEntropyLoss()
    optim_func = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = ReduceLROnPlateau(optim_func, mode='min', factor=0.1, patience=3)


    # Setting DataLoader for training and validation
    train_loader = dataloader(args=opt, mode='train')
    valid_loader = dataloader(args=opt, mode='val')


    # To store Loss and Accuracy of Training & Validation
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    best_train_acc = 0
    best_val_acc = 0

    for epoch in range(opt.epochs):
        model.train()

        train_loss, train_acc = train(
            opt, epoch, device, model, train_loader, loss_func, optim_func)
        train_loss_list.append(train_loss.detach().cpu().numpy())
        train_acc_list.append(train_acc)

        valid_loss, valid_acc = evaluate(
            opt, epoch, device, model, valid_loader, loss_func)
        valid_loss_list.append(valid_loss.detach().cpu().numpy())
        valid_acc_list.append(valid_acc)

        current_lr = optim_func.param_groups[0]['lr']

        print(
            f"Epoch {epoch + 1}/{opt.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
            f"Val Loss: {valid_loss:.4f} Acc: {valid_acc:.2%} | "
            f"LR: {current_lr:.1e}"
        )

        scheduler.step(valid_loss.item())

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'val_best.pth'))

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'train_best.pth'))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, f'epoch{epoch}.pth'))


    torch.save(model.state_dict(), os.path.join(opt.save_path, 'last.pth'))

    # Draw Figure
    draw_figure('loss', opt.save_path, opt.epochs, train_loss_list, valid_loss_list)
    print('save loss fig!')
    draw_figure('accuracy', opt.save_path, opt.epochs, train_acc_list, valid_acc_list)
    print('save acc fig!')
