# EEGNet diagnostic (exploratory; seed 0)

Reproduces P4: {"eegnet_32ch": {"identical": 62, "n": 62}, "eegnet_8ch": {"identical": 62, "n": 62}}

```
              method      protocol  n_subjects  test_ba  test_ba_p4_seed0  diff_vs_p4  p_vs_p4  train_ba  val_ba  best_epoch0_frac  best_epoch_median  n_epochs_median
eegnet_32ch_fixed100         block          22    0.337             0.350      -0.013    0.144     0.988   0.346             0.351              7.000          100.000
eegnet_32ch_fixed100 cross_session          20    0.308             0.330      -0.022    0.052     0.990   0.346             0.200             22.000          100.000
eegnet_32ch_fixed300         block          22    0.327             0.350      -0.022    0.012     1.000   0.335             0.363             10.000          300.000
eegnet_32ch_fixed300 cross_session          20    0.314             0.330      -0.016    0.185     1.000   0.354             0.200             26.000          300.000
  eegnet_32ch_logged         block          22    0.350             0.350       0.000    1.000     0.639   0.364             0.387              6.000           57.000
  eegnet_32ch_logged cross_session          20    0.330             0.330       0.000    1.000     0.762   0.372             0.200             26.000           77.000
   eegnet_32ch_val03         block          22    0.339             0.350      -0.011    0.082     0.650   0.360             0.399              6.000           57.000
   eegnet_32ch_val03 cross_session          20    0.327             0.330      -0.003    0.970     0.781   0.373             0.150             22.000           73.000
 eegnet_8ch_fixed100         block          22    0.345             0.350      -0.004    0.566     0.945   0.339             0.351             15.000          100.000
 eegnet_8ch_fixed100 cross_session          20    0.340             0.330       0.010    0.370     0.942   0.364             0.350             14.500          100.000
 eegnet_8ch_fixed300         block          22    0.337             0.350      -0.012    0.237     0.999   0.355             0.280             17.000          300.000
 eegnet_8ch_fixed300 cross_session          20    0.326             0.330      -0.004    0.526     0.999   0.373             0.400             22.000          300.000
   eegnet_8ch_logged         block          22    0.350             0.350       0.000    1.000     0.648   0.373             0.369             12.000           63.000
   eegnet_8ch_logged cross_session          20    0.330             0.330       0.000    1.000     0.638   0.407             0.350             14.500           65.500
    eegnet_8ch_val03         block          22    0.350             0.350       0.000    0.924     0.659   0.373             0.310             14.000           65.000
    eegnet_8ch_val03 cross_session          20    0.343             0.330       0.013    0.117     0.676   0.369             0.250             19.000           70.000
```
