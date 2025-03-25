import os
import csv
import torch
import argparse
import warnings
import torch.nn as nn
import torchvision.models as models

from tqdm import tqdm
from model import model
from utils import tqdm_bar
from dataloader import dataloader

cls = [str(i) for i in range(0, 100)]
cls = sorted(cls)
class_to_idx = {idx: name for idx, name in enumerate(cls)}

# ignore warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(
        description='Train the ClassificationModel')
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
    parser.add_argument('--weights', '-w', type=str, default='./best.pth')
    parser.add_argument('--batch_size', '-b', type=int, default='1')

    return parser.parse_args()

def make_csv(predition):
    with open('./prediction.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # First Row
        writer.writerow(['image_name', 'pred_label'])

        for row in predition:
            writer.writerow(row)
    print('Save prediction.csv !!!')

def test(args, model, TestLoader):
    prediction = []
    with torch.no_grad():
        for imgs, filenames in (pbar := tqdm(TestLoader, ncols=120)):

            # Set image to the same device as the model
            imgs = imgs.to(device=args.device)

            # Input the image to model
            pred = model(imgs)

            # Get the top 1 predictions
            pred = torch.argmax(pred, dim=1)

            tqdm_bar('Test', pbar)

            for filename, predict in zip(filenames, pred.cpu().tolist()):
                filename = os.path.splitext(filename)[0]
                prediction.append((filename, class_to_idx[predict]))

    return prediction


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model().to(device)

    # setting Test dataLoader
    TestLoader = dataloader(args, 'test')

    # Load trained weights
    model.load_state_dict(torch.load(args.weights))
    
    model.eval()
    prediction = test(args, model, TestLoader)

    # Saved the prediction result to csv file
    make_csv(prediction)
