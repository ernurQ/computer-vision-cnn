import torch


def evaluate_model(model, loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            output = model(images)
            loss = criterion(output, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total  # Validation accuracy
    avg_loss = val_loss / len(loader)  # Average validation loss
    return avg_loss, accuracy
