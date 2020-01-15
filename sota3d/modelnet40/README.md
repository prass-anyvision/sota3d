Description
-----------

This benchmark uses the ModelNet40 subset from the Princeton Modelnet dataset.
Here we focus on the classification task.

**Metrics**: overall accuracy and mean accuray.

Leaderboard
-----------

| Method           | overall acc.  | mean acc.     | # runs   |
| ---------------- | ------------- | ------------- | -------- |
| PointNet         | 0.883 ± 0.002 | 0.841 ± 0.006 | 5        |
| PointCNN         | 0.900 ± 0.006 | 0.864 ± 0.011 | 5        |
| PointNet++ (SSG) | 0.901 ± 0.003 | 0.860 ± 0.006 | 5        |
| DGCNN            | 0.911 ± 0.002 | 0.872 ± 0.004 | 5        |
| PointConv        | 0.905 ± 0.003 | 0.868 ± 0.004 | 5        |

Last update: 15 January 2020
