"""
Step 6c: label-free signals that can see the F1 dips.

Per 512-flow window we compute signals from the model's OUTPUTS only (no labels):
  neg_conf_pred_attack : minus the mean confidence of flows the model calls "attack"
  uncertain_share      : share of flows with 0.1 < p < 0.9
  mean_entropy         : mean binary entropy of p
  neg_mean_conf        : minus the window's mean confidence (the Step 5 definition)
  neg_pred_attack_share: minus the share of flows called "attack" (proxy for attack share)
plus, for comparison, the paper's signals from Step 6 (paper_S, paper_neg_dc, paper_delta)
and the TRUE attack share (reference only: it needs labels).
Higher = "worse" for every signal.

"Bad window" (ground truth, used only for scoring): centred rolling-mean F1 (20 windows) < BAD_F1.

Part 1: how well does each signal separate bad from good windows (AUC; 0.5 = useless, 1.0 = perfect).
Part 2: a detector on each signal, threshold fitted on VALIDATION only (mean + K*std), persistence 3,
        cooldown 20 as in the paper. Counts triggers, false triggers, bad stretches caught and lead time.

Reads : step2_data.pkl, step4_results.pkl, (optional) step6_drift.pkl
Writes: step6c_signals.pkl, step6c_signals.png        Needs: scikit-learn, matplotlib
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

FOLDER = os.environ.get("UNSW_FOLDER", r"C:\Users\Lenovo\Documents\pbl\dataset one")
WINDOW = 512
BAD_F1 = 0.90          # centred rolling F1 below this = bad stretch
F1_ROLL = 20
MIN_BAD_LEN = 10       # ignore bad stretches shorter than this (windows)
SMOOTH = 10            # trailing rolling mean applied to signals before thresholding
K = 2.0                # threshold = mean + K*std of the signal on validation
PERSIST, COOLDOWN = 3, 20
NEAR = 30              # a trigger up to this many windows before a bad stretch still counts as a catch
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

d2 = pd.read_pickle(os.path.join(FOLDER, "step2_data.pkl"))
d4 = pd.read_pickle(os.path.join(FOLDER, "step4_results.pkl"))
p6 = os.path.join(FOLDER, "step6_drift.pkl")
paper = pd.read_pickle(p6)["signals"] if os.path.exists(p6) else None
if paper is None:
    print("note: step6_drift.pkl not found, paper signals skipped")


def window_signals(split):
    y = d2["y"][split].astype(bool)
    prob = d4["prob"][split].astype("float64")
    n_w = len(y) // WINDOW
    m = n_w * WINDOW
    y, prob = y[:m].reshape(n_w, WINDOW), prob[:m].reshape(n_w, WINDOW)
    pred = prob > 0.5
    conf = np.maximum(prob, 1 - prob)
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    n_pred = pred.sum(1)
    cpa = np.where(n_pred > 0, (conf * pred).sum(1) / np.maximum(n_pred, 1), np.nan)
    tp, fp, fn = (y & pred).sum(1), (~y & pred).sum(1), (y & ~pred).sum(1)
    den = 2 * tp + fp + fn
    df = pd.DataFrame({
        "neg_conf_pred_attack": -pd.Series(cpa).ffill().fillna(1.0).values,
        "uncertain_share": ((prob > 0.1) & (prob < 0.9)).mean(1),
        "mean_entropy": ent.mean(1),
        "neg_mean_conf": -conf.mean(1),
        "neg_pred_attack_share": -pred.mean(1),
        "f1": np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan),
        "neg_true_attack_share (needs labels)": -y.mean(1),
    })
    return df


val, stream = window_signals("val"), window_signals("stream")
n_ref, n_str = len(val), len(stream)
if paper is not None:
    pv = paper.reset_index(drop=True)
    assert len(pv) == n_ref + n_str, "step6_drift.pkl does not match step5/step4 window counts"
    for name, col, sgn in [("paper_S", "S", 1), ("paper_neg_dc", "dc", -1), ("paper_delta", "delta", 1)]:
        v = pd.Series(sgn * pv[col].values.astype("float64")).bfill().fillna(0.0).values  # window 0 has no predecessor
        val[name], stream[name] = v[:n_ref], v[n_ref:]
signals = [c for c in stream.columns if c != "f1"]

# ---- ground truth: bad stretches on the stream (and on val for the AUC) ----
def bad_mask(f1):
    c = pd.Series(f1).rolling(F1_ROLL, center=True, min_periods=5).mean().values
    bad = c < BAD_F1
    # drop short runs
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a < MIN_BAD_LEN:
            bad[a:b] = False
    return bad


bad_v, bad_s = bad_mask(val["f1"].values), bad_mask(stream["f1"].values)
edges = np.flatnonzero(np.diff(np.r_[0, bad_s.astype(int), 0]))
segs = [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2] - 1)]
print(f"validation: {bad_v.sum()} bad windows of {n_ref}; stream: {bad_s.sum()} bad windows of {n_str}")
print("bad stretches on the stream (first, last window):", segs)

# ---- Part 1: separation ----
full = pd.concat([val, stream], ignore_index=True)
smooth = full[signals].rolling(SMOOTH, min_periods=1).mean()
rows = []
for s in signals:
    sv, ss = smooth[s].values[:n_ref], smooth[s].values[n_ref:]
    rows.append({"signal": s,
                 "AUC val (raw)": roc_auc_score(bad_v, val[s]) if bad_v.any() else np.nan,
                 "AUC stream (raw)": roc_auc_score(bad_s, stream[s]),
                 "AUC val (smoothed)": roc_auc_score(bad_v, sv) if bad_v.any() else np.nan,
                 "AUC stream (smoothed)": roc_auc_score(bad_s, ss)})
auc = pd.DataFrame(rows)
print("\n=== Part 1: separation of bad vs good windows (AUC) ===")
print(auc.round(3).to_string(index=False))


# ---- Part 2: detectors ----
def trigger(flag, start_at):
    held = pd.Series(flag).rolling(PERSIST).sum().fillna(0).values == PERSIST
    out, last = np.zeros(len(flag), bool), -10**9
    for t in range(start_at, len(flag)):
        if held[t] and t - last >= COOLDOWN:
            out[t], last = True, t
    return out


rows, trig_store = [], {}
for s in signals:
    x = smooth[s].values
    thr = x[:n_ref].mean() + K * x[:n_ref].std()
    flag = x > thr
    trig = np.flatnonzero(trigger(flag, n_ref)) - n_ref          # stream-local indices
    trig_store[s] = (thr, trig)
    near = np.zeros(n_str, bool)
    for a, b in segs:
        near[max(a - NEAR, 0):b + 1] = True
    false_t = [t for t in trig if not near[t]]
    caught, leads = 0, []
    for a, b in segs:
        hit = [t for t in trig if a - NEAR <= t <= b]
        if hit:
            caught += 1
            leads.append(hit[0] - a)
    rows.append({"signal": s, "triggers": len(trig), "false triggers": len(false_t),
                 f"stretches caught (of {len(segs)})": caught,
                 "lead vs stretch start (windows; <0 = early)": leads if leads else "-",
                 "flagged on val": f"{flag[:n_ref].mean():.1%}",
                 "flagged on stream": f"{flag[n_ref:].mean():.1%}"})
det = pd.DataFrame(rows)
print(f"\n=== Part 2: detectors (threshold = val mean + {K}*std of the {SMOOTH}-window smoothed signal; "
      f"persistence {PERSIST}, cooldown {COOLDOWN}) ===")
print(det.to_string(index=False))

# ---- plot ----
show = [s for s in ["neg_conf_pred_attack", "uncertain_share", "paper_S"] if s in signals]
fig, ax = plt.subplots(len(show) + 1, 1, figsize=(12, 3 * (len(show) + 1)), sharex=True)
x = np.arange(n_str)
ax[0].plot(x, stream["f1"], ".", ms=2, alpha=0.4)
ax[0].plot(x, pd.Series(stream["f1"]).rolling(F1_ROLL, center=True, min_periods=5).mean(), lw=1.2)
ax[0].set_ylabel("window F1")
for a, name in zip(ax[1:], show):
    a.plot(x, smooth[name].values[n_ref:], lw=0.9)
    a.axhline(trig_store[name][0], color="r", ls="--", lw=0.8)
    for t in trig_store[name][1]:
        a.axvline(t, color="red", alpha=0.6, lw=1)
    a.set_ylabel(name, fontsize=8)
for a in ax:
    for s0, s1 in segs:
        a.axvspan(s0, s1, color="grey", alpha=0.2)
ax[-1].set_xlabel("stream window index (grey = bad stretch; red lines = triggers)")
plt.tight_layout()
plt.savefig(os.path.join(FOLDER, "step6c_signals.png"), dpi=110)

pd.to_pickle({"auc": auc, "detectors": det, "segments": segs, "signals_val": val, "signals_stream": stream},
             os.path.join(FOLDER, "step6c_signals.pkl"))
print("\nsaved step6c_signals.pkl and step6c_signals.png")
plt.show()