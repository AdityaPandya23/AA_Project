# NSGA-II chooser rules (block, viewing period)

Exploratory (CONTRACTS k12, ANALYSIS_PLAN §5). Subject-level means (seeds, then sessions). `0sd` = best inner BA; `1sd` = contract rule (= stored ga_nsga); gap = inner-CV BA - outer BA. Choosing a rule by its outer score would be selection on the test set.

```
        method    rule  n_subjects  outer_ba_mean  outer_ba_std  inner_ba_mean  gap_mean  cost_mean  n_ch_mean
  ga_nsga_view   0.5sd          22          0.414         0.055          0.589     0.175      0.387      3.275
  ga_nsga_view     0sd          22          0.421         0.063          0.601     0.180      0.454      3.633
  ga_nsga_view     1sd          22          0.410         0.048          0.567     0.157      0.333      2.941
  ga_nsga_view     2sd          22          0.393         0.052          0.532     0.139      0.272      2.483
  ga_nsga_view    knee          22          0.398         0.049          0.551     0.153      0.288      2.629
  ga_nsga_view mincost          22          0.377         0.043          0.486     0.110      0.214      2.000
  ga_nsga_view      se          22          0.413         0.055          0.587     0.174      0.381      3.235
  ga_nsga_view  stored          22          0.410         0.048          0.567     0.157      0.333      2.941
ga_single_view  stored          22          0.428         0.074          0.625     0.198      0.539      4.100
```
