# Figure captions

**F01** (`F01_method_comparison_block.png`). **Subject-level balanced accuracy, block protocol (n = 22).** One dot per subject (seeds, then sessions averaged); lines connect a subject across methods; dashed line = chance (1/3). Means: FBCSP all 0.353, GA-NSGA (proposed) 0.345, GA single-obj. 0.353, EEGNet (32 ch) 0.341, EEGNet (8 ch) 0.347. Pre-registered primary test GA-NSGA > FBCSP all: Δ -0.007, one-sided p = 0.788 (not supported).

**F02** (`F02_cost_vs_accuracy_block.png`). **Accuracy against cost, block protocol.** Grey: every inner-CV Pareto entry of the GA (inner BA, optimistic); coloured: outer-test subject means. GA-NSGA reaches FBCSP-all accuracy at about 30 % of the cost (0.30); EEGNet has no cost (shown at n/a).

**F03** (`F03_method_comparison_cross_session.png`). **Cross-session balanced accuracy** (train ses-01, test ses-02; one dot per subject, n = 20; sub-09/10 have no ses-02; per-subject values in `tables/per_subject_cross_session.csv`). Means: FBCSP all 0.337, GA-NSGA (proposed) 0.340, GA single-obj. 0.328, EEGNet (32 ch) 0.325, EEGNet (8 ch) 0.334. sub-01 is the only subject with the same trial sequence in both sessions (see F08).

**F04** (`F04_selection_frequency_ga_nsga.png`). **How often GA-NSGA selects each gene** (block, 504 chosen chromosomes). Selection is nearly flat (18–43 %); the most frequent gene is Pz (43 %).

**F05** (`F05_train_inner_test.png`). **Training vs inner-CV vs outer-test BA** (block, n = 22; P5 §5.1, exploratory). Every model fits its training trials (FBCSP all 0.999 with 168 features on about 90 trials) but generalises to about chance + 0.02. The GA's inner-CV score of its chosen chromosome is inflated by selection (winner's curse); FBCSP all's unselected inner CV is not.

**F06** (`F06_random_control.png`). **Random-chromosome control at matched cost** (P5 §2, exploratory). Left: the GA's percentile among 50 random chromosomes with the same channel/band/window counts, per block fold; the flat histogram at the dashed 10 % line means the GA's pick is a typical random subset. Right: per subject, GA BA vs the mean BA of the random subsets (diagonal = equal).

**F07** (`F07_chooser_rules_block.png`). **Alternative chooser rules on the stored GA-NSGA fronts** (block, n = 22; P5 §3, exploratory). No rule differs from FBCSP all or from the contract rule (1sd). These are test-set scores and must not be used to pick a rule.

**F08** (`F08_position_control_cross_session.png`). **Position-only control, cross-session** (P5 §1, exploratory). A classifier that sees only the trial's position in its block (no EEG) is above chance where 30-trial block orders repeat across sessions (x axis); the EEG methods do not follow it.

**F09** (`F09_random_vs_block.png`). **Random 5-fold vs leave-one-block-out** (subject level, seeds 0–2; P5 §4.1, exploratory). No method gains from random splits (numbers = mean random − block).

**F10** (`F10_eegnet_learning_curves.png`). **EEGNet learning curves without early stopping** (300 epochs, block folds, seed 0; P5 §5.2, exploratory). Training BA reaches 1.0 while validation BA stays at chance.

**F11** (`F11_viewing_vs_imagery.png`). **Imagery (3 s) vs viewing period (3–7 s), block protocol** (n = 22; P5 §4.2, exploratory). All methods decode the viewing period; there GA-NSGA is 0.025 below FBCSP all (p = 0.004).
