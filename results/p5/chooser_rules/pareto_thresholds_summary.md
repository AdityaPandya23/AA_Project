# NSGA-II chooser rules (block)

Exploratory (CONTRACTS k12, ANALYSIS_PLAN §5). Subject-level means (seeds, then sessions). `0sd` = best inner BA; `1sd` = contract rule (= stored ga_nsga); gap = inner-CV BA - outer BA. Choosing a rule by its outer score would be selection on the test set.

```
   method    rule  fbcsp_all_mean  n_subjects  outer_ba_mean  outer_ba_sd  inner_ba_mean  gap_mean  cost_mean  n_ch_mean  kuncheva  jaccard  vs_fbcsp_diff  vs_fbcsp_p  vs_fbcsp_wins  vs_1sd_diff  vs_1sd_p
  ga_nsga   0.5sd           0.353          22          0.348        0.023          0.533     0.185      0.345      2.953     0.076    0.264         -0.004       0.633             10        0.003     0.149
  ga_nsga     0sd           0.353          22          0.351        0.018          0.545     0.194      0.415      3.366     0.066    0.309         -0.002       0.874              9        0.005     0.248
  ga_nsga     1sd           0.353          22          0.345        0.025          0.512     0.167      0.295      2.648     0.078    0.229         -0.007       0.443             10          NaN       NaN
  ga_nsga     2sd           0.353          22          0.338        0.026          0.486     0.147      0.247      2.275     0.077    0.200         -0.014       0.276              8       -0.007     0.140
  ga_nsga    knee           0.353          22          0.347        0.019          0.510     0.162      0.282      2.527     0.070    0.220         -0.005       0.614             11        0.002     0.874
  ga_nsga mincost           0.353          22          0.343        0.024          0.463     0.119      0.214      2.000     0.074    0.180         -0.009       0.503             11       -0.002     0.733
  ga_nsga      se           0.353          22          0.344        0.022          0.529     0.186      0.334      2.888     0.074    0.254         -0.009       0.289              9       -0.002     0.656
ga_single  stored           0.353          22          0.353        0.029          0.568     0.215      0.476      3.778     0.084    0.374          0.001       0.799             11        0.008     0.256
```
