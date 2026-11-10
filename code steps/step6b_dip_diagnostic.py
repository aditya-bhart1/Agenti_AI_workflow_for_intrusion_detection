"""
Step 6b: what are the F1 dips made of?

Splits every 512-flow window into regimes and compares them:
  low-attack  : attack share < LOW_SHARE   (the quiet phases before each burst)
  high-attack : attack share >= LOW_SHARE
  dip         : window whose rolling-mean F1 (ROLL windows) is below DIP_F1
Reports, per regime: F1, precision, recall, false positives and misses per window,
false-alarm rate, confidence, which attack types are missed, and what the false alarms are.

Reads : step2_data.pkl, step4_results.pkl      (nothing is retrained, no GPU needed)
Writes: step6b_diagnostic.pkl
"""
import os
import numpy as np
import pandas as pd

FOLDER = os.environ.get("UNSW_FOLDER", r"C:\Users\Lenovo\Documents\pbl\dataset one")
WINDOW = 512
LOW_SHARE = 0.12
DIP_F1 = 0.90
ROLL = 20
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

d2 = pd.read_pickle(os.path.join(FOLDER, "step2_data.pkl"))
d4 = pd.read_pickle(os.path.join(FOLDER, "step4_results.pkl"))


def f1_of(tp, fp, fn):
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else np.nan


def analyse(split):
    y = d2["y"][split].astype(bool)
    meta = d2["meta"][split]
    prob = d4["prob"][split]
    n_w = len(y) // WINDOW
    m = n_w * WINDOW
    y, prob, meta = y[:m], prob[:m], meta.iloc[:m].reset_index(drop=True)
    pred = prob > 0.5
    conf = np.maximum(prob, 1 - prob)
    wid = np.repeat(np.arange(n_w), WINDOW)

    fl = pd.DataFrame({"w": wid, "y": y, "p": pred, "conf": conf,
                       "cat": meta["attack_cat"].astype(str).values,
                       "proto_service": meta["service"].astype(str).values})
    fl["tp"] = fl.y & fl.p
    fl["fp"] = ~fl.y & fl.p
    fl["fn"] = fl.y & ~fl.p
    g = fl.groupby("w")
    wt = pd.DataFrame({"attacks": g.y.sum(), "tp": g.tp.sum(), "fp": g.fp.sum(), "fn": g.fn.sum(),
                       "conf": g.conf.mean()})
    wt["share"] = wt.attacks / WINDOW
    wt["f1"] = [f1_of(a, b, c) for a, b, c in zip(wt.tp, wt.fp, wt.fn)]
    wt["roll_f1"] = wt.f1.rolling(ROLL, min_periods=1).mean()
    wt["regime"] = np.where(wt.share < LOW_SHARE, "low-attack", "high-attack")
    wt["dip"] = wt.roll_f1 < DIP_F1
    fl["regime"] = fl.w.map(wt.regime)
    fl["dip"] = fl.w.map(wt.dip)
    return fl, wt


def group_report(fl, wt, by, title):
    rows = []
    for key, w in wt.groupby(by):
        f = fl[fl[by] == key] if by in fl else fl[fl.w.isin(w.index)]
        tp, fp, fn = int(w.tp.sum()), int(w.fp.sum()), int(w.fn.sum())
        normals = (~f.y).sum()
        rows.append({by: key, "windows": len(w), "attack share": w.share.mean(),
                     "mean window F1": w.f1.mean(),
                     "pooled precision": tp / max(tp + fp, 1), "pooled recall": tp / max(tp + fn, 1),
                     "FP / window": w.fp.mean(), "FN / window": w.fn.mean(),
                     "false-alarm rate": fp / max(normals, 1),
                     "conf (all)": f.conf.mean(),
                     "conf (normal)": f.loc[~f.y, "conf"].mean(),
                     "conf (attack)": f.loc[f.y, "conf"].mean()})
    print(f"\n--- {title} ---")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


out = {}
for split in ["val", "stream"]:
    fl, wt = analyse(split)
    out[split] = {"windows": wt}
    print(f"\n==================== {split}: {len(wt)} windows ====================")
    print(f"windows with low attack share (<{LOW_SHARE}): {(wt.regime == 'low-attack').mean():.1%}; "
          f"windows inside an F1 dip (rolling F1 < {DIP_F1}): {wt.dip.sum()}")

    group_report(fl, wt, "regime", "by attack-share regime")
    group_report(fl, wt, "dip", "dip vs not-dip")

    # is a dip a false-alarm problem or a miss problem?
    dd = wt[wt.dip]
    if len(dd):
        tot = dd.fp.sum() + dd.fn.sum()
        print(f"\nInside dips: {int(dd.fp.sum())} false alarms vs {int(dd.fn.sum())} misses "
              f"-> {dd.fp.sum() / max(tot, 1):.0%} of all errors are false alarms")

    # does F1 drop only because attacks are rare? (same errors, different attack share)
    for r, w in wt.groupby("regime"):
        print(f"{r:12s} mean F1 {w.f1.mean():.4f}   mean recall {w.tp.sum() / max(w.tp.sum() + w.fn.sum(), 1):.4f}   "
              f"mean false-alarm rate {w.fp.sum() / max((WINDOW * len(w) - w.attacks.sum()), 1):.4f}")

    # per attack type: recall in dip vs outside
    a = fl[fl.y]
    t = a.groupby(["cat", "dip"]).p.mean().unstack()
    t.columns = ["recall outside dips" if not c else "recall inside dips" for c in t.columns]
    t["rows inside dips"] = a[a.dip].groupby("cat").size()
    print("\nrecall per attack type, inside vs outside dips:")
    print(t.round(3).to_string())

    # what do the false alarms look like?
    fa = fl[fl.fp]
    print("\nfalse alarms by service (inside dips vs outside), as share of that group's false alarms:")
    s = pd.DataFrame({"inside dips": fa[fa.dip].proto_service.value_counts(normalize=True),
                      "outside dips": fa[~fa.dip].proto_service.value_counts(normalize=True)}).fillna(0)
    print(s.round(3).head(8).to_string())
    nm = fl[~fl.y]
    print("share of all NORMAL flows by service in dips (for comparison):")
    print(nm[nm.dip].proto_service.value_counts(normalize=True).round(3).head(8).to_string())

    # dips: where are they in time, and what precedes them
    dips = wt.dip.values.astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, dips, 0]))
    segs = [(edges[i], edges[i + 1] - 1) for i in range(0, len(edges), 2)]
    print("\ndip segments (first window, last window, length, mean attack share, min rolling F1):")
    for s0, s1 in segs:
        seg = wt.iloc[s0:s1 + 1]
        print(f"  {s0:>5d} - {s1:>5d}  len {len(seg):>4d}  share {seg.share.mean():.3f}  min rollF1 {seg.roll_f1.min():.3f}")

pd.to_pickle(out, os.path.join(FOLDER, "step6b_diagnostic.pkl"))
print("\nsaved step6b_diagnostic.pkl")
