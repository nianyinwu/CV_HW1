import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim


from tqdm import tqdm
from model import model
from eval import evaluate
from torch.backends import cudnn
from dataloader import dataloader
from utils import tqdm_bar, DrawFigure
from torch.optim.lr_scheduler import ReduceLROnPlateau

cudnn.benchmark = True  # fast training


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the Classification Model(ResNeXt50)')
    parser.add_argument(
        '--device',
        type=str,
        choices=[
            "cuda",
            "cpu"],
        default="cuda")
    parser.add_argument(
        '--data_path',
        '-d',
        type=str,
        default='./data',
        help='path of the input data')
    parser.add_argument(
        '--save_path',
        '-s',
        type=str,
        default='./n_saved_model',
        help='path to save the training model')
    parser.add_argument(
        '--epochs',
        '-e',
        type=int,
        default=50,
        help='number of epochs')
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=16,
        help='batch size')
    parser.add_argument(
        '--learning_rate',
        '-lr',
        type=float,
        default=1e-4,
        help='learning rate')


    return parser.parse_args()


def train(
        args,
        epoch,
        device,
        model,
        TrainLoader,
        criterion,
        optimizer):
    totalLoss = 0
    correct = 0
    for img, label in (pbar := tqdm(TrainLoader, ncols=120)):
        # Set image and label to the same device as the model
        img = img.to(device=device)
        label = label.to(device=device)

        # clear gradient
        optimizer.zero_grad()

        # input the data to model
        pred = model(img)
        # Loss Function
        loss = criterion(pred, label)

        tqdm_bar('Train', pbar, loss.detach().cpu(), epoch, args.epochs)

        # Calculate Total Loss
        totalLoss += loss

        # Calculate Top1 Accuracy
        pred = torch.argmax(pred, dim=1)
        correct += (pred == label).sum().item()

        # Backward
        loss.backward()

        # Update all parameters
        optimizer.step()

        # Clear Gradient
        optimizer.zero_grad()

    # Calculate Average Loss of current epoch
    AvgLoss = totalLoss / len(TrainLoader)
    AvgAcc = correct / len(TrainLoader.dataset)
    return AvgLoss, AvgAcc


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    model = model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)


    # setting Dataset and DataLoader
    TrainLoader = dataloader(args=args, mode='train')
    ValidLoader = dataloader(args=args, mode='val')

    # To store Loss and Accuracy of Training & Validation
    Train_loss = []
    Train_accuracy = []
    Valid_loss = []
    Valid_accuracy = []
    train_best = 0
    val_best = 0

    for epoch in range(args.epochs):
        model.train()

        train_loss, train_accuracy = train(
            args, epoch, device, model, TrainLoader, criterion, optimizer)
        Train_loss.append(train_loss.detach().cpu().numpy())
        Train_accuracy.append(train_accuracy)

        val_loss, val_accuracy = evaluate(
            args, epoch, device, model, ValidLoader, criterion)
        Valid_loss.append(val_loss.detach().cpu().numpy())
        Valid_accuracy.append(val_accuracy)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_accuracy:.2%} | Val Loss: {val_loss:.4f} Acc: {val_accuracy:.2%} | LR: {current_lr:.1e}")

        scheduler.step(val_loss.item())

        if val_accuracy > val_best:
            val_best = val_accuracy
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_path,
                    'val_best.pth'))

        if train_accuracy > train_best:
            train_best = train_accuracy
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_path,
                    'train_best.pth'))

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_path,
                    f'epoch{epoch}.pth'))


    torch.save(model.state_dict(), os.path.join(args.save_path, 'last.pth'))

    DrawFigure('loss', args.save_path, args.epochs, Train_loss, Valid_loss)
    print('save loss fig!')
    DrawFigure('accuracy', args.save_path, args.epochs, Train_accuracy, Valid_accuracy)
    print('save acc fig!')

