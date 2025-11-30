import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import random


# seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# device
def set_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="fcnn", choices=["linear", "fcnn", "cnn"]
    )
    parser.add_argument(
        "-o", "--optimizer", type=str, default="adamw", choices=["sgd", "adamw"]
    )
    parser.add_argument(
        "-s", "--scheduler", type=str, default="cos", choices=["step", "cos", "cosine"]
    )
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-lr", "--lr", type=float, default=1e-3)
    parser.add_argument("-bs", "--batch_size", type=int, default=128)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


# data
def data_process(batch_size, train=False, test=False):
    transform = {
        "train": transforms.Compose(
            [
                transforms.RandomCrop(size=(32, 32), padding=1),
                # transforms.RandomResizedCrop(
                #     size=(32, 32), scale=(0.8, 1.0), ratio=(0.5, 2.0)
                # ),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                # transforms.ColorJitter(
                #     brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                # ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    }

    loader = {}

    if train:
        train_dataset_full = datasets.CIFAR10(
            root="./data", download=True, train=True, transform=transform["train"]
        )
        val_dataset_full = datasets.CIFAR10(
            root="./data", download=True, train=True, transform=transform["test"]
        )

        full_size = len(train_dataset_full)
        train_size = int(0.9 * full_size)
        indices = list(range(full_size))
        np.random.shuffle(indices)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

        train_dataset = Subset(train_dataset_full, train_idx)
        val_dataset = Subset(val_dataset_full, val_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        loader["train"] = train_loader
        loader["val"] = val_loader

    if test:
        test_dataset = datasets.CIFAR10(
            root="./data", download=True, train=False, transform=transform["test"]
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        loader["test"] = test_loader

    return loader


# model
class LinearClassifier(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(c_in, c_out)

    def forward(self, x):
        return self.fc(self.flatten(x))


class FCNN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out):
        super().__init__()

        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(c_in, c_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(c_hidden, c_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(c_hidden // 2, c_hidden // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(c_hidden // 4, c_out),
        )

    def forward(self, x):
        return self.net(self.flatten(x))


class CNN(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, c_out)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def set_model(args):
    name = args.model
    if name == "linear":
        model = LinearClassifier(in_channels * h * w, out_channels).to(device)
    elif name == "fcnn":
        model = FCNN(in_channels * h * w, hidden_channels, out_channels).to(device)
    else:
        model = CNN(in_channels, out_channels).to(device)

    print(
        f"model={args.model}, optim={args.optimizer}, sched={args.scheduler}, lr={args.lr}, epochs={args.epochs}"
    )

    return model


# optimizer
def set_optimizer(args):
    name = args.optimizer
    if name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    return optimizer


# scheduler
def set_scheduler(args):
    name = args.scheduler
    if name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    return scheduler


# train
def train(model, optimizer, scheduler, args):
    print("\n----------Start Training----------")

    # epochs
    epochs = args.epochs

    # data
    loader = data_process(args.batch_size, train=True)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # checkpoint
    ckpt_path = f"checkpoints/{args.model}_{args.optimizer}_{args.scheduler}_lr{args.lr}_epochs{args.epochs}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0

    # tensorboard
    writer_name = f"runs/{args.model}_{args.optimizer}_{args.scheduler}_lr{args.lr}"
    with SummaryWriter(writer_name) as writer:

        for epoch in range(epochs):
            print(f"\nEpoch: {epoch}")

            # loss & acc
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            # train
            model.train()
            for images, labels in loader["train"]:
                # device
                images = images.to(device)
                labels = labels.to(device)

                # optimizer
                optimizer.zero_grad()

                # forward
                output = model(images)
                loss = criterion(output, labels)

                # backward
                loss.backward()

                # gradient descent
                optimizer.step()

                # loss & acc
                train_loss += loss.item() * labels.shape[0]
                _, predicted = torch.max(output, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.shape[0]

            # val
            with torch.no_grad():
                model.eval()
                for images, labels in loader["val"]:
                    images = images.to(device)
                    labels = labels.to(device)
                    output = model(images)
                    loss = criterion(output, labels)

                    val_loss += loss.item() * labels.shape[0]
                    _, predicted = torch.max(output, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.shape[0]

            # scheduler
            scheduler.step()

            # loss & acc
            train_loss /= train_total
            val_loss /= val_total
            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")

            writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"Train": train_acc, "Val": val_acc}, epoch)

            # checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                # print(f"Model saved to {ckpt_path}")

    return


# test
def test(model, args):
    print("\n----------Start Testing----------")

    # data
    loader = data_process(args.batch_size, test=True)

    # load model
    ckpt_path = f"checkpoints/{args.model}_{args.optimizer}_{args.scheduler}_lr{args.lr}_epochs{args.epochs}.pt"
    if not os.path.exists(ckpt_path):
        print("Model hasn't been trained")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    test_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        model.eval()
        for images, labels in loader["test"]:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)

            test_loss += loss.item() * labels.shape[0]
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.shape[0]

    print(f"Test  Loss: {test_loss/total:.4f}, Test  Acc: {correct/total*100:.2f}%")

    return


# main
if __name__ == "__main__":

    # hyper-parameters
    in_channels = 3
    h = w = 32
    hidden_channels = 1024
    out_channels = 10

    # seed
    set_seed(42)

    # argparse
    args = get_args()

    # device
    device = set_device()

    # model
    model = set_model(args)

    # train
    if args.train:
        # optimizer
        optimizer = set_optimizer(args)

        # scheduler
        scheduler = set_scheduler(args)

        train(model, optimizer, scheduler, args)

    # test
    if args.test:
        test(model, args)
