import os
import yaml
import tqdm
import torch
import torch.nn as nn
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        config,
        dataloaders,
        criteria,
        metrics,
        optimizer,
        scheduler,
        device="cuda",
        logdir="logs",
        log_interval=10,
        epochs=100,
        monitor="off",
        early_stopping=False,
        tensorboard=False,
        tensorboard_cb=None,
        eval_freq=1,
        resume=False,
    ):
        assert "train" in dataloaders
        self.model = model
        self.config = config
        self.dataloaders = dataloaders
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        if monitor == "off":
            self.mode = "off"
            self.monitor = None
            self.best = 0
        else:
            self.mode, self.monitor = monitor.split()
            self.best = np.inf if self.mode == "min" else -np.inf
        self.device = device
        self.epochs = epochs
        self.logdir = logdir
        self.log_interval = log_interval
        self.writer = None
        self.tensorboard_cb = tensorboard_cb
        self.eval_freq = eval_freq
        self.initial_epoch = 1
        self.early_stopping = early_stopping
        self.resume = resume

    def fit(self):
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        fname = os.path.join(self.logdir, "config.yaml")
        with open(fname, "w") as fp:
            yaml.dump(self.config, fp, default_flow_style=False)

        self.model.to(self.device)
        if self.resume:
            self._load_checkpoint(best=False)
            print("Resume from epoch {:03d}".format(self.initial_epoch))

        for epoch in range(self.initial_epoch, self.epochs + 1):
            stats = self._train_epoch(epoch)
            if epoch % self.eval_freq == 0:
                phase = "train"
                if "val" in self.dataloaders:
                    phase = "val"
                elif "test" in self.dataloaders:
                    phase = "test"
                stats.update(self._eval_epoch(phase))

            self._save_checkpoint(epoch, best=False)
            if epoch % self.eval_freq == 0:
                best = False
                if self.mode != "off":
                    curr = stats[self.monitor]
                    if self.mode == "min":
                        best = curr < self.best
                    elif self.mode == "max":
                        best = curr > self.best
                if best:
                    print("Saving checkpoint...")
                    self.best = curr
                    self._save_checkpoint(epoch, best)

    def evaluate(self, phase="test"):
        self.model.to(self.device)
        self._load_checkpoint(best=True)
        self._eval_epoch(phase)

    def predict(self, phase="test"):
        self.model.to(self.device)
        self._load_checkpoint(best=True)

        self.model.eval()
        with torch.no_grad():
            for i, (inputs, target) in enumerate(self.dataloaders[phase]):
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                output = self.model(inputs)
                yield output.cpu()

    def _train_epoch(self, epoch, phase="train"):
        meter = AverageMeter()

        postfix = {}
        desc = "Epoch [{:03d}/{:03d}]".format(epoch, self.epochs)
        pbar = tqdm.tqdm(total=len(self.dataloaders[phase]), desc=desc)
        postfix["lr"] = self._get_lr()

        if self.metrics is not None:
            for metric in self.metrics:
                metric.reset()

        self.model.train()
        for i, (inputs, target) in enumerate(self.dataloaders[phase]):
            inputs = [x.to(self.device) for x in inputs]
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(*inputs)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()

            meter.update(loss.item())
            postfix["loss"] = meter.avg

            if self.metrics is not None:
                for metric in self.metrics:
                    metric.update(output, target)
                    postfix[metric.name] = metric.score()

            # TODO: tensorboard logging
            if self.writer is not None and i % self.log_interval == 0:
                pass

            pbar.set_postfix(**postfix)
            pbar.update()
        pbar.close()

        if self.scheduler is not None:
            self.scheduler.step()
        return postfix

    def _eval_epoch(self, phase):
        meter = AverageMeter()

        postfix = {}
        desc = "Evaluation"
        pbar = tqdm.tqdm(total=len(self.dataloaders[phase]), desc=desc)

        if self.metrics is not None:
            for metric in self.metrics:
                metric.reset()

        self.model.eval()
        with torch.no_grad():
            for i, (inputs, target) in enumerate(self.dataloaders[phase]):
                inputs = [x.to(self.device) for x in inputs]
                target = target.to(self.device)

                output = self.model(*inputs)
                loss = self.criteria(output, target)

                meter.update(loss.item())

                if self.metrics is not None:
                    for metric in self.metrics:
                        metric.update(output, target)
                        postfix[metric.name] = metric.score()

                if self.writer is not None and i % self.log_interval == 0:
                    pass

                postfix["loss"] = meter.avg
                pbar.set_postfix(**postfix)
                pbar.update()
            pbar.close()
        return postfix

    def _save_checkpoint(self, epoch, best=False):
        checkpoint = {
            "epoch": epoch,
            "best": self.best,
            "config": self.config,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        basename = "best.pth" if best else "last.pth"
        fname = os.path.join(self.logdir, basename)
        torch.save(checkpoint, fname)

    def _load_checkpoint(self, best=False):
        basename = "best.pth" if best else "last.pth"
        fname = os.path.join(self.logdir, basename)
        state = torch.load(fname)
        self.initial_epoch = state["epoch"] + 1
        self.best = state["best"]
        self.model.load_state_dict(state["model"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])

    def _get_lr(self):
        for param in self.optimizer.param_groups:
            return param["lr"]


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def update(self, val, step=1):
        self.val = val
        self.total += val * step
        self.count += step
        self.avg = self.total / self.count
