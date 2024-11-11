import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import csv


def get_model(model_path: str = None):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_classes = 36

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model


def save_model(model: nn.Module, epoch: int):
    save_dir = os.path.join('checkpoints', 'dropout')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch}.pth'))


def save_accuracy(accuracy: float, epoch: int):
    save_dir = os.path.join('accuracy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file = os.path.join(save_dir, 'dropout.csv')
    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, accuracy])
