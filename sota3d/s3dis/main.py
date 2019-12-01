import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch3d.models as models
import torch3d.datasets as datasets
import torch3d.metrics as metrics
import torch3d.transforms as transforms

from .. import Trainer
from .. import utils


def create_transform(config):
    if config["model"]["name"] == "pointnet":
        transform = transforms.ToTensor()
    elif config["model"]["name"] == "pointnet2":

        def transform(pcd):
            return pcd[..., :3], pcd[..., 3:].T

    elif config["model"]["name"] == "dgcnn":
        transform = transforms.ToTensor()

    return transform


def create_dataloaders(config, transform):
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            datasets.S3DIS(
                config["dataset"]["root"],
                train=True,
                test_area=config["dataset"]["test_area"],
                transform=transform,
                download=config["dataset"]["download"],
            ),
            batch_size=config["dataset"]["batch_size"],
            num_workers=config["dataset"]["num_workers"],
            pin_memory=True,
            shuffle=True,
        ),
        "test": torch.utils.data.DataLoader(
            datasets.S3DIS(
                config["dataset"]["root"],
                train=False,
                test_area=config["dataset"]["test_area"],
                transform=transform,
                download=False,
            ),
            batch_size=config["dataset"]["batch_size"],
            num_workers=config["dataset"]["num_workers"],
            pin_memory=True,
            shuffle=False,
        ),
    }
    return dataloaders


def create_model(config):
    model = None
    if config["model"]["name"] == "pointnet":
        model = models.segmentation.PointNet(
            config["model"]["in_channels"], config["model"]["num_classes"]
        )
    elif config["model"]["name"] == "pointnet2":
        model = models.segmentation.PointNetSSG(
            config["model"]["in_channels"], config["model"]["num_classes"]
        )
    elif config["model"]["name"] == "dgcnn":
        model = models.segmentation.DGCNN(
            config["model"]["in_channels"], config["model"]["num_classes"]
        )
    return model


def create_optimizer(config, parameters):
    optimizer = None
    if config["optimizer"]["name"] == "sgd":
        optimizer = optim.SGD(
            parameters,
            lr=config["optimizer"]["lr"],
            momentum=config["optimizer"]["momentum"],
        )
    elif config["optimizer"]["name"] == "adam":
        optimizer = optim.Adam(parameters, lr=config["optimizer"]["lr"])
    return optimizer


def create_metrics(config):
    accuracy = metrics.Accuracy(config["model"]["num_classes"])
    jaccard = metrics.Jaccard(config["model"]["num_classes"])
    return [accuracy, jaccard]


def main(args, options=None):
    with open(args.config, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)
    if options:
        config = utils.override_config(config, options)
    print(yaml.dump(config))

    transform = create_transform(config)
    dataloaders = create_dataloaders(config, transform)
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
