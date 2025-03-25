import torch
from tqdm import tqdm
from utils import tqdm_bar


def evaluate(args, epoch, device, model, ValidLoader, criterion):

    # to let Batch Normalization and Dropout be closed
    model.eval()

    TotalLoss = 0
    correct = 0

    # torch.np_grad to stop autograd
    with torch.no_grad():
        for img, label in (pbar := tqdm(ValidLoader, ncols=120)):

            # Set image and label to the same device as the model
            img = img.to(device=device)
            label = label.to(device=device)

            # input the data to model
            pred = model(img)

            # Loss Function
            Loss = criterion(pred, label)

            # Calculate Top1 Accuracy
            pred = torch.argmax(pred, dim=1)
            correct += (pred == label).sum().item()

            tqdm_bar('Val', pbar, Loss.detach().cpu(), epoch, args.epochs)

            # Calculate total Loss and total dice score
            TotalLoss += Loss

        AvgLoss = TotalLoss / len(ValidLoader)
        AvgAcc = correct / len(ValidLoader.dataset)

    return AvgLoss, AvgAcc
