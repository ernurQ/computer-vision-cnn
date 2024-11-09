import torch.optim as optim
import torch.nn as nn
from datetime import datetime

from cnn_model import CNNModel, ActivationFunction, InitMethod
from data_loader import train_loader, validation_loader
from evaluate import evaluate_model


def _train_one_epoch(model: CNNModel, loader, criterion, optimizer):
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()  # Clear the gradients
        output = model(images)  # Forward pass
        loss = criterion(output, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights


def _graceful_shutdown():
    # Flag for controlling the training loop
    global training
    training = True

    # Custom handler for graceful shutdown
    def signal_handler(sig, frame):
        global training
        training = False

    # Register the signal handler for graceful interruption
    import signal
    signal.signal(signal.SIGINT, signal_handler)


def train(
        activation_function: ActivationFunction,
        init_method: InitMethod,
        epoch=0,
        model_path=None,
):
    _graceful_shutdown()
    model = CNNModel(activation_function, init_method)
    if model_path is not None:
        model.load(model_path)

    criterion = nn.CrossEntropyLoss()
    l2_lambda = 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_lambda)

    prev_validation_loss = 0.

    print(f'Start training model: {model.name}')
    while True:
        epoch += 1
        _train_one_epoch(model, train_loader, criterion, optimizer)
        validation_loss, accuracy = evaluate_model(model, validation_loader, criterion)
        loss_change = validation_loss - prev_validation_loss
        prev_validation_loss = validation_loss

        model.save(epoch)

        print(f'Epoch {epoch}, Validation Loss: {validation_loss:.4f}, Accuracy: {accuracy:.2f}%')
        print(f'Validation loss change: {loss_change}')
        print(datetime.now())
        print('--------------------------------------------------')
