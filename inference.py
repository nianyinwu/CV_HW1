"""Inference script for image classification."""

import os
import csv
import argparse
import warnings

from typing import List, Tuple

import torch
from tqdm import tqdm

from model import get_model
from utils import tqdm_bar
from dataloader import dataloader




# Class ID mapping
class_names = sorted([str(i) for i in range(100)])
class_to_idx = {i: name for i, name in enumerate(class_names)}


# ignore warnings
warnings.filterwarnings('ignore')

def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument(
        '--device',
        type=str,
        choices=["cuda", "cpu"],
        default="cuda"
    )
    parser.add_argument(
        '--data_path',
        '-d',
        type=str,
        default='./data',
        help='Path to input data'
    )
    parser.add_argument(
        '--weights',
        '-w',
        type=str,
        default='./best.pth',
        help='Path to model weights'
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=1,
        help='Batch size for inference'
    )

    return parser.parse_args()

def make_csv(predictions: list[tuple[str, str]]) -> None:
    """
    Generate prediction CSV file.
    """

    with open(file='./prediction.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'pred_label'])  # header
        writer.writerows(predictions)
    print('Save prediction.csv !!!')


def test(
    args: argparse.Namespace,
    test_model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader
) -> List[Tuple[str, str]]:
    """
    Perform inference on the test set.

    Returns:
        List of tuples (image_name, predicted_class)
    """
    predictions = []
    test_model.eval()

    with torch.no_grad():
        for imgs, filenames in (pbar := tqdm(data_loader, ncols=120)):
            # Set image to the same device as the model
            imgs = imgs.to(args.device)

            # Input the image to model
            outputs = test_model(imgs)

            # Get the top 1 prediction
            preds = torch.argmax(outputs, dim=1)

            tqdm_bar('Test', pbar)

            for filename, pred in zip(filenames, preds.cpu().tolist()):
                image_name = os.path.splitext(filename)[0]
                predictions.append((image_name, class_to_idx[pred]))

    return predictions


if __name__ == "__main__":
    opt = get_args()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = get_model().to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    model.eval()

    # Load test data
    test_loader = dataloader(opt, mode='test')

    # Inference
    preditions = test(opt, model, test_loader)

    # Save to CSV
    make_csv(preditions)
