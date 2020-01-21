Description
-----------

This benchmark uses the S3DIS dataset for semantic segmentation task. Here we
use k-fold validation.

**Metrics**: overall accuracy, mean accuracy, and mean IoU.

Leaderboard
-----------

| Method           | overall acc.  | mean acc.     | mean IoU      |
| ---------------- | ------------- | ------------- | ------------- |
| PointNet         | 0.827 ± 0.051 | 0.599 ± 0.120 | 0.496 ± 0.120 |
| PointNet++ (SSG) | 0.868 ± 0.022 | 0.694 ± 0.093 | 0.580 ± 0.100 |
| DGCNN            | 0.870 ± 0.044 | 0.691 ± 0.116 | 0.586 ± 0.129 |

Last update: 21 January 2020
