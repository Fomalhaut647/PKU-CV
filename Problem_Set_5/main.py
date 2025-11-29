import torch
import torch.nn as nn
import argparse


class LinearClassifier(nn.Module):
    # define a linear classifier
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # inchannels: dimension of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimension=3072
        # out_channels: number of categories. For CIFAR-10, it's 10

    def forward(self, x: torch.Tensor):
        return


class FCNN(nn.Module):
    # def a full-connected neural network classifier
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        # inchannels: dimension of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimension=3072
        # hidden_channels
        # out_channels: number of categories. For CIFAR-10, it's 10

        # full connected layer
        # activation function
        # full connected layer
        # ......

    def forward(self, x: torch.Tensor):
        return


def train(model, optimizer, scheduler, args):
    """
    Model training function
    input:
        model: linear classifier or full-connected neural network classifier
        loss_function: Cross-entropy loss
        optimizer: Adamw or SGD
        scheduler: step or cosine
        args: configuration
    """
    # create dataset

    # create dataloader

    # for-loop
    # train
    # get the inputs; data is a list of [inputs, labels]

    # zero the parameter gradients

    # forward

    # loss backward

    # optimize

    # adjust learning rate

    # test
    # forward
    # calculate accuracy

    # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)


def test(model, args):
    """
    input:
        model: linear classifier or full-connected neural network classifier
        loss_function: Cross-entropy loss
    """
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    # create testing dataset
    # create dataloader
    # test
    # forward
    # calculate accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="The configs")

    parser.add_argument("--run", type=str, default="train")
    parser.add_argument("--model", type=str, default="linear")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--scheduler", type=str, default="step")
    args = parser.parse_args()

    # create model
    if args.model == "linear":
        model = None
    elif args.model == "fcnn":
        model = None
    else:
        raise AssertionError

    # create optimizer
    if args.optimizer == "adamw":
        # create Adamw optimizer
        optimizer = None
    elif args.optimizer == "sgd":
        # create SGD optimizer
        optimizer = None
    else:
        raise AssertionError

    # create scheduler
    if args.scheduler == "step":
        # create torch.optim.lr_scheduler.StepLR scheduler
        scheduler = None
    elif args.scheduler == "cosine":
        # create torch.optim.lr_scheduler.CosineAnnealingLR scheduler
        scheduler = None
    else:
        raise AssertionError

    if args.run == "train":
        train(model, optimizer, scheduler, args)
    elif args.run == "test":
        test(model, args)
    else:
        raise AssertionError

# You need to implement training and testing function that can choose model, optimizer, scheduler and so on by command, such as:
# python main.py --run=train --model=fcnn --optimizer=adamw --scheduler=step
