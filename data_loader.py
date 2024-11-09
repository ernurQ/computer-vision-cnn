import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Data transformations
_train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3,
                           contrast=0.3,
                           saturation=0.3,
                           hue=0.1),
    transforms.RandomAffine(degrees=0,
                            translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset (replace with the path to your test dataset)
_root_folder = f'{os.getcwd()}/data'
_train_dataset = datasets.ImageFolder(root=f'{_root_folder}/train', transform=_train_transform)
_validation_dataset = datasets.ImageFolder(root=f'{_root_folder}/validation', transform=_test_transform)
_test_dataset = datasets.ImageFolder(root=f'{_root_folder}/test', transform=_test_transform)

train_loader = DataLoader(_train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(_validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(_test_dataset, batch_size=32, shuffle=False)