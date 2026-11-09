"""Every number in docs/RESULTS.md and docs/P3_REPORT_R07.md.

Usage (repo root): ``python -m reports.p3_numbers <scratch_dir>``. Copies the method
folders into ``<scratch_dir>/ana`` (``load_results("results")`` trips on
``results/sanity/sanity_summary.json``) and prints the tables.
"""
import json, shutil, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")
from src import analysis as A, chromosome  # noqa: E402

SCR = Path(sys.argv[1])
ana = SCR / "ana"
if ana.exists():
    shutil.rmtree(ana)
for m in ["ga_nsga", "ga_single", "eegnet_32ch", "eegnet_8ch"]:
    shutil.copytree(f"results/{m}", ana / m)
shutil.copytree("results/pareto", ana / "pareto")
shutil.copytree("results/baseline_all/fbcsp_all", ana / "fbcsp_all")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    df = A.load_results(ana)
print("load warnings:", [str(x.message) for x in w])
print("hashes:", df.config_hash.unique(), "files per method/protocol:")
print(df.groupby(["method", "protocol"]).size().to_string())

R07 = [f"{i:02d}" for i in range(17, 23)]
ALL = [f"{i:02d}" for i in range(1, 23)]
GA = ["fbcsp_all", "ga_single", "ga_nsga"]
EEG = ["fbcsp_all", "eegnet_32ch", "eegnet_8ch", "ga_nsga"]
f3 = lambda x: f"{x:.3f}"


def table(sub, methods, title):
    d = df[df.subject.isin(sub)]
    print(f"\n== {title}")
    for m in methods:
        dm = d[d.method == m]
        row = [m]
        for prot in ["block", "cross_session"]:
            s = A.subject_level(dm, prot)
            row.append(f"{s[m].mean():.3f} ± {s[m].std(ddof=1):.3f} (n={len(s)})" if m in s else "-")
        blk = dm[dm.protocol == "block"]
        ex = A.explode_folds(blk)
        nf = ex.n_features.dropna().astype(float).mean()
        cost = ex.cost.dropna().astype(float).mean() if ex.cost.notna().any() else np.nan
        row += [f"cost={cost:.3f}", f"feat/params={nf:.0f}", f"infer={blk.infer_time_ms.mean():.3f}ms",
                f"train={blk.train_time_s.mean():.1f}s/run"]
        ac = A.above_chance(d, m, "block")
        ac["p_holm"] = A.holm(ac.p.to_numpy())
        row.append(f"rec p<.05 {int((ac.p < .05).sum())}/{int((ac.p_holm < .05).sum())} of {len(ac)}")
        print("  ", " | ".join(row))
    return d


def pairs(d, comps, label):
    print(f"-- paired Wilcoxon two-sided ({label})")
    for a, b in comps:
        for prot in ["block", "cross_session"]:
            try:
                r = A.paired_test(d, a, b, prot)
                print(f"   {a} - {b} {prot}: n={r['n']} d={r['mean_diff']:+.3f} med={r['median_diff']:+.3f} "
                      f"p={r['p']:.3f} r_rb={r['effect_size']:+.2f}")
            except ValueError as e:
                print("  ", e)


# ------------------------------------------------------------ R07 share: GA
d7 = table(R07, GA, "R07 share sub-17..22, GA")
pairs(d7, [("ga_nsga", "fbcsp_all"), ("ga_single", "fbcsp_all"), ("ga_nsga", "ga_single")], "n=6")
for prot in ["block", "cross_session"]:
    w = A.subject_level(d7, prot)[GA]
    print(f"   per subject {prot}:\n" + w.round(3).to_string())
for m in ["ga_nsga", "ga_single"]:
    ac = A.above_chance(d7, m, "block"); acb = A.above_chance(d7, "fbcsp_all", "block")
    print(f"   sig recordings {m}:", list(ac[ac.p < .05][["subject", "session"]].itertuples(index=False, name=None)),
          "fbcsp_all:", list(acb[acb.p < .05][["subject", "session"]].itertuples(index=False, name=None)))

print("\n-- inner vs outer (chosen chromosome) R07 share")
for m in ["ga_nsga", "ga_single"]:
    for prot in ["block", "cross_session"]:
        ex = A.explode_folds(d7[(d7.method == m) & (d7.protocol == prot)])
        inner, size = [], []
        for _, r in ex.iterrows():
            e = [p for p in r.pareto if p["bits"] == r.chosen_bits]
            inner.append(e[0]["ba_mean"]); size.append(len(r.pareto))
        print(f"   {m} {prot}: inner={np.mean(inner):.3f} outer={ex.ba.mean():.3f} "
              f"optimism={np.mean(inner) - ex.ba.mean():+.3f} front={np.mean(size):.1f} folds={len(ex)}")
    blk = d7[(d7.method == m) & (d7.protocol == "block")]
    sd = blk.groupby(["subject", "session"]).ba_mean.std(ddof=1).mean()
    ex = A.explode_folds(blk)
    B = np.stack([chromosome.from_str(s) for s in ex.chosen_bits])
    print(f"   {m} seed SD per recording={sd:.3f}  mean ch/bands/win = "
          f"{B[:, :8].sum(1).mean():.1f}/{B[:, 8:12].sum(1).mean():.1f}/{B[:, 12:].sum(1).mean():.1f}")
    fr = A.selection_frequency(blk, m).sort_values("frequency", ascending=False)
    print(f"   freq range {fr.frequency.min():.2f}-{fr.frequency.max():.2f}; top5:",
          ", ".join(f"{r.name} {100 * r.frequency:.0f}%" for r in fr.head(5).itertuples()))
    for by in ["all", "seed", "fold"]:
        s = A.selection_stability(blk, m, by=by)
        print(f"   stability block by={by}: kuncheva={s.kuncheva.mean():.3f} jaccard={s.jaccard.mean():.2f}")
    s = A.selection_stability(d7[(d7.method == m) & (d7.protocol == "cross_session")], m, by="all")
    print(f"   stability cross by=all(seeds): kuncheva={s.kuncheva.mean():.3f} jaccard={s.jaccard.mean():.2f}")

# ------------------------------------------------------------ EEGNet, all 22
de = table(ALL, EEG, "EEGNet, all subjects")
pairs(de, [("eegnet_32ch", "fbcsp_all"), ("eegnet_8ch", "fbcsp_all"), ("eegnet_8ch", "eegnet_32ch")], "n=22/20")
ee = de[de.method.str.startswith("eegnet")]
for m in ["eegnet_32ch", "eegnet_8ch"]:
    rs = [json.load(open(p)) for p in ee[ee.method == m].path]
    be = [b for r in rs for b in (f["best_epoch"] for f in r["per_fold"])]
    ne = [b for r in rs for b in (f["n_epochs"] for f in r["per_fold"])]
    print(f"   {m}: n_params={rs[0]['per_fold'][0]['n_features']} best_epoch median={np.median(be):.0f} "
          f"(IQR {np.percentile(be, 25):.0f}-{np.percentile(be, 75):.0f}) n_epochs median={np.median(ne):.0f} "
          f"infer_batch1={np.mean([r['infer_time_ms_batch1'] for r in rs]):.2f}ms "
          f"seedSD={ee[ee.method == m].groupby(['subject', 'session', 'protocol']).ba_mean.std(ddof=1).mean():.3f}")
cd = A.cross_session_drop(df, ["fbcsp_all", "eegnet_32ch", "eegnet_8ch", "ga_nsga", "ga_single"])
print("-- cross-session drop (ses-02 block - cross), with / without sub-01")
for m, g in cd.groupby("method"):
    print(f"   {m}: {g['drop'].mean():+.3f} (n={len(g)}) / {g[g.subject != '01']['drop'].mean():+.3f}")

# ------------------------------------------------------------ pre-registered group analysis
print("\n== GROUP (ANALYSIS_PLAN)")
# completeness rule
exp = {}
for m in ["ga_nsga", "ga_single", "eegnet_32ch", "eegnet_8ch", "fbcsp_all"]:
    dm = df[df.method == m]
    seeds = [0] if m == "fbcsp_all" else [0, 1, 2]
    miss = []
    for s in ALL:
        sess = ["01"] if s in ("09", "10") else ["01", "02"]
        for ses in sess:
            for sd in seeds:
                if not ((dm.subject == s) & (dm.session == ses) & (dm.seed == sd) & (dm.protocol == "block")).any():
                    miss.append((s, ses, sd))
        if s not in ("09", "10"):
            for sd in seeds:
                if not ((dm.subject == s) & (dm.protocol == "cross_session") & (dm.seed == sd)).any():
                    miss.append((s, "01to02", sd))
    print(f"   completeness {m}: missing {len(miss)} {miss[:5]}")

wide = A.subject_level(df, "block")
d = (wide.ga_nsga - wide.fbcsp_all).dropna()
prim = A.paired_test(df, "ga_nsga", "fbcsp_all", "block")
one = stats.wilcoxon(d.to_numpy(), zero_method="wilcox", alternative="greater")
print(f"PRIMARY ga_nsga>fbcsp_all block: n={prim['n']} mean={prim['mean_diff']:+.4f} median={prim['median_diff']:+.4f} "
      f"W={prim['statistic']:.1f} one-sided p={one.pvalue:.4f} r_rb={prim['effect_size']:+.3f} "
      f"SD(d)={d.std(ddof=1):.4f}  (MDD assumed SD 0.044 -> 0.023)  wins={int((d > 0).sum())}/{len(d)}")
se, tq = d.std(ddof=1) / np.sqrt(len(d)), stats.t.ppf(0.975, len(d) - 1)
print(f"   exploratory 95% t-interval of mean diff: {d.mean() - tq * se:+.4f} .. {d.mean() + tq * se:+.4f}")
fam = [("S1", "ga_nsga", "ga_single", "block"), ("S2", "ga_nsga", "fbcsp_all", "cross_session"),
       ("S3", "ga_nsga", "eegnet_32ch", "block"), ("S4", "ga_nsga", "eegnet_8ch", "block")]
res = [A.paired_test(df, a, b, p) for _, a, b, p in fam]
ph = A.holm([r["p"] for r in res])
for (k, a, b, p), r, h in zip(fam, res, ph):
    print(f"   {k} {a} vs {b} {p}: n={r['n']} mean={r['mean_diff']:+.4f} median={r['median_diff']:+.4f} "
          f"W={r['statistic']:.1f} p={r['p']:.4f} p_holm={h:.4f} r_rb={r['effect_size']:+.3f}")
dn = df[df.subject != "01"]
r = A.paired_test(dn, "ga_nsga", "fbcsp_all", "cross_session")
print(f"   S2 without sub-01: n={r['n']} mean={r['mean_diff']:+.4f} p={r['p']:.4f} r_rb={r['effect_size']:+.3f}")

print("\n-- descriptive, all subjects")
METH = ["fbcsp_all", "ga_single", "ga_nsga", "eegnet_32ch", "eegnet_8ch"]
for m in METH:
    dm = df[df.method == m]
    out = [m]
    for prot in ["block", "cross_session"]:
        s = A.subject_level(dm, prot)[m]
        out.append(f"{prot} {s.mean():.3f} ± {s.std(ddof=1):.3f} (n={len(s)}) above-chance subj {int((s > 1/3).sum())}")
    ex = A.explode_folds(dm[dm.protocol == "block"])
    cost = ex.cost.dropna().astype(float).mean() if ex.cost.notna().any() else np.nan
    out.append(f"cost {cost:.3f} feat {ex.n_features.dropna().astype(float).mean():.1f} "
               f"infer {dm[dm.protocol == 'block'].infer_time_ms.mean():.3f}ms")
    for prot in ["block", "cross_session"]:
        ac = A.above_chance(df, m, prot)
        out.append(f"{prot} rec p<.05 {int((ac.p < .05).sum())}/{int((ac.p_holm < .05).sum())} of {len(ac)}")
    if m.startswith("ga"):
        B = np.stack([chromosome.from_str(s) for s in ex.chosen_bits])
        out.append(f"ch/b/w {B[:, :8].sum(1).mean():.1f}/{B[:, 8:12].sum(1).mean():.1f}/{B[:, 12:].sum(1).mean():.1f}")
    print("   " + " | ".join(out))
print("   friedman block:", A.friedman_test(df, METH, "block"))
for m in ["ga_nsga", "ga_single"]:
    blk = df[(df.method == m) & (df.protocol == "block")]
    fr = A.selection_frequency(blk, m).sort_values("frequency", ascending=False)
    print(f"   {m} freq {fr.frequency.min():.2f}-{fr.frequency.max():.2f} top5:",
          ", ".join(f"{r.name} {100 * r.frequency:.0f}%" for r in fr.head(5).itertuples()),
          "| bottom3:", ", ".join(f"{r.name} {100 * r.frequency:.0f}%" for r in fr.tail(3).itertuples()))
    for by in ["all", "seed", "fold"]:
        s = A.selection_stability(blk, m, by=by)
        print(f"   {m} stability by={by}: kuncheva={s.kuncheva.mean():.3f} jaccard={s.jaccard.mean():.2f}")
    s = A.selection_stability(df[(df.method == m) & (df.protocol == "cross_session")], m, by="all")
    print(f"   {m} stability cross (seeds): kuncheva={s.kuncheva.mean():.3f} jaccard={s.jaccard.mean():.2f}")
    for prot in ["block", "cross_session"]:
        ex = A.explode_folds(df[(df.method == m) & (df.protocol == prot)])
        inner = [[p for p in r.pareto if p["bits"] == r.chosen_bits][0]["ba_mean"] for r in ex.itertuples()]
        print(f"   {m} {prot} inner {np.mean(inner):.3f} outer {ex.ba.mean():.3f} optimism {np.mean(inner) - ex.ba.mean():+.3f}")
wide.round(4).to_csv(SCR / "subject_block.csv")
A.subject_level(df, "cross_session").round(4).to_csv(SCR / "subject_cross.csv")
print("\nblock subject table:\n" + wide[METH].round(3).to_string())
print("\ncross subject table:\n" + A.subject_level(df, "cross_session")[METH].round(3).to_string())
