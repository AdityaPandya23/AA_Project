# P2 review: FBCSP baseline and sanity checks

Reviewer: R01, 2026-11-08. Reviewed: `docs/P2_REPORT.md` (R02) and the committed
results in `results/baseline_all/` and `results/sanity/` (code `4f6109f`, config
`08e124167b`).

**Verdict: the P2 report is correct and its results reproduce exactly.** Every
number was recomputed independently from the committed JSONs, and a full re-run
on R01 gives bit-identical results. The comments below are for precision and
interpretation. None of them changes a conclusion.

## 1. Reproduction

On R01 (Windows 11, Python 3.11.9) with `main` @ `372c407`, using the 42-recording
cache from R02 (`check_manifest --subjects all`: OK):

```
python -m experiments.baseline --n-jobs -1 --out <tmp>/baseline_all   # 49 s
python -m experiments.sanity  --n-jobs -1 --out <tmp>/sanity          # 3.5 min
```

| Output | Files | Identical `ba_mean` | Identical per-trial `y_pred` |
|---|---|---|---|
| `baseline_all/fbcsp_all` | 62 | 62 / 62 | 62 / 62 |
| `sanity/` (random splits + viewing) | 252 | 252 / 252 | 252 / 252 |
| `sanity_summary.json` (a, b, c) | – | identical, including the shuffled-label null | – |

So the pipeline is deterministic across machines, given the same cache. A cache
built independently on R01 from the raw data (sub-09..16) differs from R02's
only by floating-point rounding in `X` (at most 1e-6 µV, about 2e-8 of the signal;
labels, blocks and rejections identical). That cannot change a prediction in
practice, but is the reason to share one cache for P3.

## 2. Numbers in the report, recomputed

All values use `src.analysis` on the committed JSONs (1000 permutations, one-sided).

| Report claim | Recomputed | OK |
|---|---|---|
| Block per recording: 0.356 ± 0.060, median 0.350, range 0.233–0.476, n = 42 | 0.356 ± 0.061 (sample SD), 0.350, 0.233–0.476 | ✅ see §3.1 |
| 10 / 2 recordings above chance (uncorrected / Holm) | 10 / 2: 02/01, 02/02, 03/01, 05/01, 11/02, 13/02, 14/02, 20/02, 22/01, 22/02; Holm: 02/02, 20/02 (both 0.475) | ✅ |
| Per subject 0.353 ± 0.043, Wilcoxon > 1/3 p = 0.027 (n = 22) | 0.353 ± 0.044, p = 0.027 (one-sample t-test p = 0.028) | ✅ |
| ses-01 0.349 vs ses-02 0.364 | 0.349 (n = 22) vs 0.364 (n = 20) | ✅ |
| Cross-session 0.337 ± 0.038, median 0.336, range 0.270–0.406, n = 20; 2 / 0 above chance | 0.337 ± 0.039, 0.336, 0.270–0.406; 2 / 0 (sub-01 p = 0.043, sub-05 p = 0.049) | ✅ |
| Without sub-01: 0.333 (n = 19) | 0.3332 | ✅ |
| Within 0.360 vs cross 0.337, drop 0.023 | 0.360 is the **sessions-averaged** within-session BA; see §3.2 | ✅ |
| (a) Shuffled null 0.333 ± 0.050; 6/42 at p < 0.05 (refit) | identical | ✅ |
| (b) Random 0.353 vs block 0.356, inflation −0.004, 18/42 higher | identical (−0.0035) | ✅ |
| (c) Viewing 0.438 ± 0.075 vs 0.356, +0.082, 32/42 higher | 0.438 ± 0.076, +0.082, 32/42 | ✅ |
| sub-01: best cross-session result, within-session 0.300 and 0.361 | rank 1 of 20 (0.406); 0.300 / 0.361 | ✅ |

## 3. Points to fix or clarify

1. **SD convention (fixed in code by this PR).** The report and
   `sanity_summary` give SDs across recordings with `ddof=0`, while
   `reports/make_tables.py` uses the sample SD (`ddof=1`). The two differ by
   ≤ 0.001 here, but the paper tables and the text would disagree in the third
   decimal. `experiments/sanity.py` and `experiments/baseline.py` (R01) now use
   `ddof=1` as well. The committed `sanity_summary.*` are left as R02 produced
   them; they pick up the new convention on the next run. *Suggest:* R02 updates
   the four SDs in `P2_REPORT.md` (0.061, 0.044, 0.039, 0.076) or adds "population SD".
2. **Definition of the cross-session drop.** The report's 0.023 compares
   cross-session BA with the within-session BA **averaged over both sessions**.
   `make_tables` defines the drop as **ses-02 block BA − cross-session BA**
   (0.027 ± 0.085), because both are then scored on the same ses-02 trials. Both
   are legitimate, but the paper should use one. *Proposal:* ses-02, as in
   `make_tables` and `analysis.cross_session_drop`'s default; R02 to confirm.
3. **"10 of 42 above chance" as group evidence.** Under the null, ≥ 10 of 42 at
   α = 0.05 has binomial probability 3e-5. That assumes independent
   recordings, but the two sessions of a subject are not independent, and 4 of the
   10 come from just two subjects (sub-02 and sub-22, both sessions each). The subject-level Wilcoxon
   (p = 0.027) is the right group test and is what the report leads with. Keep it
   that way.
4. **Viewing window (DATA_NOTES open point).** Viewing being decodable (§2c)
   shows that the 3–7 s window contains perceptual signal, so the epoching is not
   grossly misaligned. It does not by itself rule out the alternative in
   DATA_NOTES (picture appearing at t ≈ 1 s): a sustained response would also
   make 3–7 s decodable. The imagery alignment rests on the evoked transients in
   DATA_NOTES §3, which are unaffected. *Suggest:* "largely settles" →
   "supports"; optionally a one-off check of BA for windows starting at t = 1, 2
   and 3 s on a few subjects if the paper describes the viewing timing.

## 4. Additional results worth reporting

| Test | Result |
|---|---|
| Cross-session, group level (Wilcoxon > 1/3, n = 20) | p = 0.41: no transfer at group level |
| Viewing vs imagery, subject level (Wilcoxon, n = 22) | p = 3e-4, r_rb = 0.87 (`make_tables`, Holm p < 0.001) |
| Viewing above chance per recording (uncorrected / Holm) | 25 / 12 of 42 |
| Random vs block splits (paired Wilcoxon over recordings) | p = 0.72: no detectable inflation |
| Confusion matrices (block, pooled) | Imagery: diagonal 0.35 / 0.34 / 0.38 (circle / square / pentagram). Viewing: 0.41 / 0.40 / **0.50**; circle and square are confused mostly with each other (0.34 both ways). Pentagram is the most separable shape in both periods. |

## 5. Implications for P3

* The bar to beat is weak: subject-level block BA 0.353, i.e. +0.020 above
  chance, with an SD of 0.044 over subjects. A GA gain of a few points would be a
  large relative effect, but with n = 22 subjects a paired test will only
  detect differences of roughly ≥ 0.02–0.03 BA reliably. (80 % power, one-sided
  α = 0.05, if the SD of the paired GA − baseline differences is about the
  between-subject SD of 0.044: Δ ≈ 2.49 × 0.044 / √22 ≈ 0.023. The real SD of
  the differences is only known after P3.) The planned primary
  comparison (`ga_nsga` vs `fbcsp_all`, block, subject level) should be stated
  before P3 results are in, with `ga_single` and cross-session as secondary.
* Cross-session is at chance for the baseline. GA cross-session results above
  chance should be read together with the sub-01 flag (§3 of the report) and the
  position-only control (R07, P5).
* Re-check the random-vs-block comparison for the GA and EEGNet, as the report
  says: a selection that favours slow, drifting features would show up there first.

## 6. Tables and figures produced (this PR)

From `results/baseline_all` + `results/sanity` with
`python -m reports.make_tables --results results/baseline_all results/sanity --reference fbcsp_all`
and `python -m src.analysis --results results/baseline_all results/sanity --out results/figures`:

* `results/tables/main_results.{md,csv}`, `results/tables/per_subject.csv`
* `results/figures/subject_bars_{block,cross_session,random}.png`,
  `method_comparison_block.png`, `confusion_fbcsp_all_block.png`,
  `confusion_fbcsp_all_view_block.png`, `cost_vs_accuracy_block.png`
  (only one cost point for now; it becomes informative with the GA results).

Plot fixes made while reviewing (R01 code):

* `_view` variants shared the exact colour of their base method, so imagery and
  viewing bars were indistinguishable. They now use a lighter tint of the same hue
  with hatching (`analysis.method_fill`).
* Per-subject bars had a one-group empty margin on both sides; removed.
* `_view` variants are left out of the cost-vs-accuracy plot (different task
  period, not a cheaper selection).

## 7. Actions

| Who | What |
|---|---|
| R02 | §3.1 SD wording, §3.2 drop definition, §3.4 wording in `P2_REPORT.md` |
| R02 / all | Agree on the P3 primary comparison (§5) before results come in |
| R07 | Position-only control and with/without sub-01 (already assigned for P5) |
| R01 | Done in this PR: SD convention, figures, tables, this review |
