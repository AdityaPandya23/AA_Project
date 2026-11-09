# Training vs inner-CV vs test BA (exploratory)

Subject level (folds, seeds, then sessions averaged). train = resubstitution on the outer training set; inner = GA fitness of the chosen chromosome (fbcsp_all: all-on, 3-fold inner CV, seed 0).

```
   method      protocol  n_subjects  train_ba_mean  train_ba_std  inner_ba_mean  test_ba_mean  test_ba_std  train_minus_test  inner_minus_test  n_features_mean
fbcsp_all         block          22          0.999         0.001          0.342         0.353        0.044             0.647            -0.010          168.000
fbcsp_all cross_session          20          0.997         0.005          0.361         0.337        0.039             0.660             0.024          168.000
  ga_nsga         block          22          0.576         0.022          0.512         0.345        0.025             0.231             0.167           13.250
  ga_nsga cross_session          20          0.547         0.042          0.491         0.340        0.027             0.207             0.151           12.900
ga_single         block          22          0.751         0.048          0.568         0.353        0.029             0.398             0.215           38.341
ga_single cross_session          20          0.753         0.078          0.552         0.328        0.027             0.425             0.224           47.200
```
