# NSGA-II chooser rules (cross_session)

Exploratory (CONTRACTS k12, ANALYSIS_PLAN §5). Subject-level means (seeds, then sessions). `0sd` = best inner BA; `1sd` = contract rule (= stored ga_nsga); gap = inner-CV BA - outer BA. Choosing a rule by its outer score would be selection on the test set.

```
   method    rule  fbcsp_all_mean  n_subjects  outer_ba_mean  outer_ba_sd  inner_ba_mean  gap_mean  cost_mean  n_ch_mean  kuncheva  jaccard  vs_fbcsp_diff  vs_fbcsp_p  vs_fbcsp_wins  vs_1sd_diff  vs_1sd_p
  ga_nsga   0.5sd           0.337          20          0.336        0.021          0.512     0.176      0.356      3.033     0.078    0.281         -0.001       0.856             11       -0.004     0.507
  ga_nsga     0sd           0.337          20          0.338        0.019          0.525     0.187      0.460      3.600     0.101    0.365          0.001       0.872             10       -0.002     0.784
  ga_nsga     1sd           0.337          20          0.340        0.027          0.491     0.151      0.301      2.800     0.051    0.225          0.003       0.601             11          NaN       NaN
  ga_nsga     2sd           0.337          20          0.341        0.026          0.464     0.123      0.240      2.233     0.108    0.220          0.004       0.546             11        0.000     0.948
  ga_nsga    knee           0.337          20          0.334        0.024          0.489     0.155      0.273      2.483     0.083    0.217         -0.003       0.904             12       -0.006     0.245
  ga_nsga mincost           0.337          20          0.342        0.027          0.449     0.107      0.214      2.000     0.071    0.183          0.005       0.654             10        0.002     0.879
  ga_nsga      se           0.337          20          0.338        0.023          0.509     0.171      0.345      2.967     0.084    0.278          0.001       0.723             12       -0.002     0.698
ga_single  stored           0.337          20          0.328        0.027          0.552     0.224      0.518      3.917     0.278    0.532         -0.009       0.351              8       -0.012     0.277
```
