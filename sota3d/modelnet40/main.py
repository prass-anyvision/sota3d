import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch3d
import torch3d.datasets as datasets
import torch3d.transforms as transforms
import torch3d.metrics as metrics

from .. import Trainer
from .. import utils


def create_transform(config):
    if config['model']['name'] == 'pointnet':
        transform = transforms.ToTensor()
    elif config['model']['name'] == 'pointcnn':
        transform = transforms.Compose([
            transforms.Downsample(1024),
            transforms.Shuffle(),
            transforms.ToTensor()
        ])
    return transform


def create_dataloaders(config, transform):
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            datasets.ModelNet40(
                config['dataset']['root'],
                train=True,
                transform=transform,
                download=config['dataset']['download']
            ),
            batch_size=config['dataset']['batch_size'],
            num_workers=config['dataset']['num_workers'],
            pin_memory=True,
            shuffle=True
        ),
        'test': torch.utils.data.DataLoader(
            datasets.ModelNet40(
                config['dataset']['root'],
                train=False,
                transform=transforms.ToTensor(),
                download=False
            ),
            batch_size=config['dataset']['batch_size'],
            num_workers=config['dataset']['num_workers'],
            pin_memory=True,
            shuffle=False
        )
    }
    return dataloaders


def create_model(config):
    model = None
    if config['model']['name'] == 'pointnet':
        model = torch3d.models.PointNet(
            config['model']['in_channels'],
            config['model']['num_classes']
        )
    elif config['model']['name'] == 'pointcnn':
        model = torch3d.models.PointCNN(
            config['model']['in_channels'],
            config['model']['num_classes']
        )
    return model


def create_optimizer(config, parameters):
    optimizer = None
    if config['optimizer']['name'] == 'sgd':
        optimizer = optim.SGD(
            parameters,
            config['optimizer']['lr'],
            momentum=config['optimizer']['momentum']
        )
    elif config['optimizer']['name'] == 'adam':
        optimizer = optim.Adam(parameters, config['optimizer']['lr'])
    return optimizer


def create_metrics(config):
    accuracy = metrics.Accuracy(config['model']['num_classes'])
    return [accuracy]


def main(args, options=None):
    with open(args.config, 'r') as fp:
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
        **config['trainer']
    )
    if not args.eval:
        trainer.fit()
    trainer.evaluate()
    # report evaluation metrics
    categories = datasets.ModelNet40.categories
    if metrics is not None:
        for metric in metrics:
            metric.report(categories)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--eval', default=False, action='store_true')
    args, options = parser.parse_known_args()
    main(args, options)
