import matplotlib.pyplot as plt
import torch.nn as nn

from data_loader import test_loader
from evaluate import evaluate_model
from model import get_model


def _create_plot(loss, accuracy, model_name):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(['Loss', 'Accuracy'], [loss, accuracy], color=['skyblue', 'orange'])
    ax.set_title(f'{model_name} Model Evaluation')
    ax.set_ylabel('Value')

    for i, v in enumerate([loss, accuracy]):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)

    plt.show()


def test(model_path: str, model_name: str):
    model = get_model(model_path)
    criterion = nn.CrossEntropyLoss()
    loss, accuracy = evaluate_model(model, test_loader, criterion)

    _create_plot(loss, accuracy, model_name)
