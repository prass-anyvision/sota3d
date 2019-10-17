import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch3d
import torch3d.datasets as datasets
import torch3d.metrics as metrics
from torch3d.transforms import *

from .. import Trainer
from .. import utils


def create_transforms(config):
    if config["model"]["name"] == "pointnet":
        transforms = {
            "train": First(ToTensor()),
            "test": First(ToTensor())
        }
    elif config["model"]["name"] == "pointcnn":
        transforms = {
            "train": Compose([
                First(RandomDownsample(1024)),
                First(Shuffle())
            )],
            "test": None
        }
    return transforms


def create_dataloaders(config, transforms):
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            datasets.ModelNet40(
                config["dataset"]["root"],
                train=True,
                transform=transform["train"],
                download=config["dataset"]["download"]
            ),
            batch_size=config["dataset"]["batch_size"],
            num_workers=config["dataset"]["num_workers"],
            pin_memory=True,
            shuffle=True
        ),
        "test": torch.utils.data.DataLoader(
            datasets.ModelNet40(
                config["dataset"]["root"],
                train=False,
                transform=transforms["test"],
                download=False
            ),
            batch_size=config["dataset"]["batch_size"],
            num_workers=config["dataset"]["num_workers"],
            pin_memory=True,
            shuffle=False
        )
    }
    return dataloaders


def create_model(config):
    model = None
    if config["model"]["name"] == "pointnet":
        model = torch3d.models.PointNet(
            config["model"]["in_channels"],
            config["model"]["num_classes"]
        )
    elif config["model"]["name"] == "pointcnn":
        model = torch3d.models.PointCNN(
            config["model"]["in_channels"],
            config["model"]["num_classes"]
        )
    return model


def create_optimizer(config, parameters):
    optimizer = None
    if config["optimizer"]["name"] == "sgd":
        optimizer = optim.SGD(
            parameters,
            config["optimizer"]["lr"],
            momentum=config["optimizer"]["momentum"]
        )
    elif config["optimizer"]["name"] == "adam":
        optimizer = optim.Adam(parameters, config["optimizer"]["lr"])
    return optimizer


def create_metrics(config):
    return [metrics.Accuracy(config["model"]["num_classes"])]


def main(args, options=None):
    with open(args.config, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    if options:
        config = utils.override_config(config, options)
        print(yaml.dump(config))

    transforms = create_transforms(config)
    dataloaders = create_dataloaders(config, transforms)
    model = create_model(config)
    optimizer = create_optimizer(config, model.parameters())
    criteria = nn.CrossEntropyLoss()
    metrics = create_metrics(config)
    trainer = Trainer(
        model,
        config,
        dataloaders,
        criteria,
        metrics,
        optimizer,
        scheduler=None,
        **config["trainer"]
    )
    if not args.eval:
        trainer.fit()
    trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--eval", default=False, action="store_true")
    args, options = parser.parse_known_args()
    main(args, options)
