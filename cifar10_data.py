# cifar10_data.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(batch_size=128, seed=None):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size

    if seed is None:
        train_set, val_set = random_split(full_train, [train_size, val_size])
    else:
        train_set, val_set = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def get_cifar10_test_loader(batch_size=128):
    _, _, test_loader = get_cifar10_loaders(batch_size=batch_size)
    return test_loader
