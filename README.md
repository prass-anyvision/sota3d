SotA3d
======

SotA3d is an attempt to reproduce and benchmark state-of-the-art 3D deep learning methods on [PyTorch](https://pytorch.org/).

The current supported benchmarks are:

**Classification**
- [*ModelNet40*](sota3d/modelnet40): Princeton ModelNet Dataset [[URL](https://modelnet.cs.princeton.edu)]

**Semantic segmentation**
- [*S3DIS*](sota3d/s3dis): Stanford Large-Scale 3D Indoor Spaces Dataset [[URL](http://buildingparser.stanford.edu/dataset.html)]

Why SotA3d?
-----------

SotA3d has three goals:
1. Promoting **reproducible** research in 3D deep learning.
2. A standard and transparent benchmarking environment for 3D deep learning methods.
3. Sharing up-to-date state-of-the-art models in 3D deep learning.

Guidelines
----------

The evaluation process must be *deterministic*, that means the evaluation scores obtained from different runs of a method using the same parameters have to be the same.
Specifically, the following items must **NOT** depend on any random processes:
- Evaluation metrics
- Order of the test set
- Content of the input

Usage
-----

Sota3d requires PyTorch 1.2 or newer. Additional dependencies:
- [torch3d](https://github.com/pqhieu/torch3d)
- tqdm
- pyyaml

Each benchmark is structured as a Python module. To train a model on a specific
dataset, enter the following command:

```bash
python -m sota3d.<dataset>.main --config configs/<dataset>/<model>.yaml
```
