import os
from enum import Enum

import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as init
import torch

class ActivationFunction(Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"

class InitMethod(Enum):
    XAVIER = "xavier"
    KAIMING = "kaiming"
    SMALL_RANDOM = "small_random"


class CNNModel(nn.Module):
    def __init__(self, activation_function: ActivationFunction, init_method: InitMethod):
        super(CNNModel, self).__init__()

        # Convolutional layers with Batch Normalization and Dropout
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.25)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        num_classes=36
        self.fc3 = nn.Linear(256, num_classes)

        # Store the activation function and initialization method
        self.activation_function = activation_function
        self.init_method = init_method

        # Apply weight initialization
        self._initialize_weights()

        self.name = f'{self.activation_function.value}_{self.init_method.value}'

    def forward(self, x):
        # Convolutional layer sequence with batch normalization, dropout, and pooling
        x = self.pool(self._apply_activation(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(self._apply_activation(self.bn2(self.conv2(x))))
        x = self.dropout1(x)

        x = self.pool(self._apply_activation(self.bn3(self.conv3(x))))
        x = self.dropout2(x)
        x = self.pool(self._apply_activation(self.bn4(self.conv4(x))))
        x = self.dropout2(x)

        # Flatten the output for fully connected layers
        x = x.view(-1, 256 * 4 * 4)

        # Fully connected layers with dropout and activation
        x = self._apply_activation(self.fc1(x))
        x = self.dropout3(x)
        x = self._apply_activation(self.fc2(x))
        x = self.fc3(x)

        return x

    def save(self, epoch: int):
        save_dir = os.path.join('models', self.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch}.pth'))

    def load(self, model_path: str):
        self.load_state_dict(torch.load(model_path))

    def _apply_activation(self, x):
        if self.activation_function == ActivationFunction.RELU:
            return functional.relu(x)
        elif self.activation_function == ActivationFunction.LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=0.01)
        elif self.activation_function == ActivationFunction.ELU:
            return functional.elu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")

    def _initialize_weights(self):
        # Apply the selected initialization method to each layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if self.init_method == InitMethod.XAVIER:
                    init.xavier_uniform_(m.weight)
                elif self.init_method == InitMethod.KAIMING:
                    init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_method == InitMethod.SMALL_RANDOM:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    raise ValueError(f"Unsupported initialization method: {self.init_method}")

                if m.bias is not None:
                    init.constant_(m.bias, 0)
