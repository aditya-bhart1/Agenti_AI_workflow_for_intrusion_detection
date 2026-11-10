"""
Step 8: injected drift. A fair test of drift detectors and model updates, because we know when drift starts.

The stream is copied and, from window W0 on, a few features are shifted:
  sudden : full shift at W0
  gradual: shift grows linearly from 0 at W0 to full size at W0 + RAMP
The shifted features are the ones most correlated with the label in TRAIN, moved toward the "normal" side
(this is our own choice: the paper only says "perturbations"). The shift size MAG is auto-calibrated so the
no-update model loses at least DROP_TARGET F1 on the drifted calm stretch (set MAG yourself to skip that).

Part 1 calibration       : no-update F1 on [W0, DRIFT_END) for several MAG.
Part 2 detection         : paper fused rule, paper both-signals rule, our confidence rule (no-update run).
Part 3 updates           : none / detector (ours, online) / paper-trigger schedule / oracle / periodic.
Part 4                   : F1 before drift, during drift, after; plot.

Reads : step2_data.pkl, step4_model.pt, step4_results.pkl, step5_windows.pkl (not needed), plus the scripts
        step7_selective_update.py and step6_drift_signals.py in the same folder.
Writes: step8_results.pkl, step8_drift.png          Takes roughly 6-12 minutes on the GPU.
"""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import step7_selective_update as s7
import step6_drift_signals as s6

FOLDER, WINDOW, dev = s7.FOLDER, s7.WINDOW, s7.dev
W0 = int(os.environ.get("W0", 560))               # drift starts here (calm phase between the natural dips)
DRIFT_END = int(os.environ.get("DRIFT_END", 960))  # the second natural dip starts at 961
RAMP = int(os.environ.get("RAMP", 300))
NFEAT = int(os.environ.get("NFEAT", 6))
MAG = os.environ.get("MAG")                        # unset = auto-calibrate
DROP_TARGET = float(os.environ.get("DROP_TARGET", 0.05))
SEED = s7.SEED
# update strength used in this step (stronger than Step 7's default; override with env vars)
s7.UPDATE_LR = float(os.environ.get("DRIFT_LR", 5e-4))
s7.UPDATE_EPOCHS = int(os.environ.get("DRIFT_EPOCHS", 3))
s7.N_REPLAY = int(os.environ.get("DRIFT_REPLAY", 5000))
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


@torch.no_grad()
def probs_embs(model, xs, T, n_w, bs=4096):
    model.eval()
    n = n_w * WINDOW
    ps, es = [], []
    for s in range(0, n, bs):
        rows = torch.arange(s, min(s + bs, n), device=dev)
        logit, emb = model(s7.make_batch(xs, rows, T), return_embedding=True)
        ps.append(torch.sigmoid(logit)); es.append(emb)
    p = torch.cat(ps).cpu().numpy()
    e = torch.cat(es).cpu().numpy()
    return p, e.reshape(n_w, WINDOW, -1).mean(1)


def drift_matrix(X_clean, shift_vec, kind, mag):
    """X_clean: (N, F) tensor; shift_vec: (F,) signed unit shift (0 for untouched features)."""
    w = torch.arange(X_clean.shape[0], device=dev) // WINDOW
    if kind == "sudden":
        f = (w >= W0).float()
    else:
        f = ((w - W0).float() / RAMP).clamp(0, 1)
    return X_clean + (f[:, None] * mag) * shift_vec[None, :]


def pooled(y2d, pred2d, a, b):
    y, p = y2d[a:b], pred2d[a:b]
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    return {"F1": s7.f1_from(tp, fp, fn), "recall": tp / max(tp + fn, 1), "false alarm": fp / max((~y).sum(), 1)}


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    d2 = pd.read_pickle(os.path.join(FOLDER, "step2_data.pkl"))
    d4 = pd.read_pickle(os.path.join(FOLDER, "step4_results.pkl"))
    ck = torch.load(os.path.join(FOLDER, "step4_model.pt"), map_location=dev)
    T, n_feat, d, base_state = ck["T"], ck["n_feat"], ck["d"], ck["state"]
    names = d2["feature_names"]

    n_w = len(d2["y"]["stream"]) // WINDOW
    m = n_w * WINDOW
    y2d = d2["y"]["stream"][:m].astype(bool).reshape(n_w, WINDOW)
    nv = len(d2["y"]["val"]) // WINDOW
    assert W0 + 50 < DRIFT_END <= n_w, "check W0 / DRIFT_END against the number of stream windows"

    # features to drift: highest |corr| with the label in TRAIN, shifted toward the normal side
    Xtr_np, ytr = d2["X"]["train"], d2["y"]["train"].astype("float64")
    sd = Xtr_np.std(0)
    corr = np.array([np.corrcoef(Xtr_np[:, j], ytr)[0, 1] if sd[j] > 0 else 0.0 for j in range(n_feat)])
    top = np.argsort(-np.abs(corr))[:NFEAT]
    shift = np.zeros(n_feat, dtype="float32")
    shift[top] = -np.sign(corr[top])
    print("drifted features:", [(names[j], round(float(corr[j]), 2)) for j in top])
    shift_vec = torch.tensor(shift, device=dev)

    # detector threshold from VALIDATION with the original model (as in Steps 6c/7)
    val_sig = window_sig = s7.window_conf_attack(d4["prob"]["val"][:nv * WINDOW].reshape(nv, WINDOW))
    val_sig = pd.Series(val_sig).ffill().fillna(-1.0).values
    sm = pd.Series(val_sig).rolling(s7.SMOOTH, min_periods=1).mean().values
    thr = float(sm.mean() + s7.K * sm.std())

    Xs_clean = torch.tensor(d2["X"]["stream"], device=dev)
    Xval = torch.tensor(d2["X"]["val"], device=dev)
    base_ctx = {"Ys": torch.tensor(d2["y"]["stream"], dtype=torch.float32, device=dev),
                "Xtr": torch.tensor(Xtr_np, device=dev),
                "Ytr": torch.tensor(d2["y"]["train"], dtype=torch.float32, device=dev),
                "T": T, "n_w": n_w, "n_feat": n_feat, "d": d, "thr": thr,
                "val_sig_tail": list(val_sig[-(s7.SMOOTH - 1):])}

    def ctx_for(Xs):
        return {**base_ctx, "Xs": Xs}

    # ---- Part 1: calibration of MAG (sudden drift, no update) ----
    ctl_probs, _ = s7.run_stream("none", base_state, ctx_for(Xs_clean))
    ctl = pooled(y2d, ctl_probs > 0.5, W0, DRIFT_END)
    print(f"\ncontrol (no drift, no update): F1 on windows {W0}-{DRIFT_END} = {ctl['F1']:.4f}")
    if MAG is not None:
        mag = float(MAG)
        print(f"using MAG={mag} (set by you)")
    else:
        rows, mag = [], None
        for cand in [0.1, 0.2, 0.3, 0.5, 0.8]:
            pr, _ = s7.run_stream("none", base_state, ctx_for(drift_matrix(Xs_clean, shift_vec, "sudden", cand)))
            f = pooled(y2d, pr > 0.5, W0, DRIFT_END)
            rows.append({"MAG": cand, "F1 in drift range": f["F1"], "drop vs control": ctl["F1"] - f["F1"],
                         "recall": f["recall"], "false alarm": f["false alarm"]})
            if mag is None and ctl["F1"] - f["F1"] >= DROP_TARGET:
                mag = cand
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        if mag is None:
            mag = 0.8
            print("WARNING: no MAG reached the target drop; using 0.8 (drift may be too weak to matter)")
        print(f"chosen MAG = {mag}")

    results, wf1s, upd, detect = [], {}, {}, []
    base_model = s7.Backbone(n_feat, T, d).to(dev)
    base_model.load_state_dict(copy.deepcopy(base_state))
    pv, ev = probs_embs(base_model, Xval, T, nv)
    conf_v = np.maximum(pv, 1 - pv).reshape(nv, WINDOW).mean(1)

    for kind in ["sudden", "gradual"]:
        print(f"\n################ {kind} drift (MAG {mag}) ################")
        Xs = drift_matrix(Xs_clean, shift_vec, kind, mag)
        ctx = ctx_for(Xs)

        # Part 2: detection on the no-update run
        p0, e0 = probs_embs(base_model, Xs, T, n_w)
        pred0 = (p0 > 0.5).reshape(n_w, WINDOW)
        conf_s = np.maximum(p0, 1 - p0).reshape(n_w, WINDOW).mean(1)
        sig = s6.run_detector(np.vstack([ev, e0]), np.concatenate([conf_v, conf_s]), nv)
        paper_fused = np.flatnonzero(sig["trigger_fused"].values[nv:])
        paper_both = np.flatnonzero(sig["trigger_both"].values[nv:])
        det = s7.DetectorState(base_ctx["val_sig_tail"], thr)
        s_arr = s7.window_conf_attack(p0.reshape(n_w, WINDOW))
        ours = np.array([t for t in range(n_w) if det.step(t, s_arr[t])])
        for nm, tr in [("paper fused rule", paper_fused), ("paper both-signals rule", paper_both),
                       ("our confidence rule", ours)]:
            after = [int(t) for t in tr if t >= W0]
            detect.append({"drift": kind, "detector": nm, "triggers before W0": [int(t) for t in tr if t < W0],
                           "first trigger at/after W0": after[0] if after else "-",
                           "delay (windows)": (after[0] - W0) if after else "-",
                           "triggers in drift range": len([t for t in after if t < DRIFT_END])})
        print(pd.DataFrame([r for r in detect if r["drift"] == kind]).to_string(index=False))

        # Part 3: updates
        paper_sched = {int(t) for t in paper_fused if t >= W0}
        oracle_at = {min(W0 + s7.LABEL_DELAY + 5, n_w - 1)} if kind == "sudden" else \
                    {min(W0 + s7.LABEL_DELAY + 5 + 60 * i, n_w - 1) for i in range(5)}
        runs = {"none": ("none", None), "detector (ours)": ("detector", None),
                "paper-trigger schedule": ("oracle", paper_sched), "oracle": ("oracle", oracle_at),
                "periodic": ("periodic", None)}
        for name, (mode, at) in runs.items():
            if mode == "oracle" and not at:
                results.append({"drift": kind, "run": name, "updates": 0}); continue
            probs, updates = (p0.reshape(n_w, WINDOW), []) if name == "none" else \
                s7.run_stream(mode, base_state, ctx, at)
            pred = probs > 0.5
            tp, fp, fn = s7.window_counts(y2d, pred)
            wf1s[(kind, name)] = np.array([s7.f1_from(a, b, c) for a, b, c in zip(tp, fp, fn)])
            upd[(kind, name)] = updates
            pre, dr, post = pooled(y2d, pred, 0, W0), pooled(y2d, pred, W0, DRIFT_END), pooled(y2d, pred, DRIFT_END, n_w)
            results.append({"drift": kind, "run": name, "updates": len(updates),
                            "F1 before drift": pre["F1"], "F1 drift range": dr["F1"],
                            "recall drift range": dr["recall"], "false alarm drift range": dr["false alarm"],
                            "F1 after": post["F1"], "update windows": updates if len(updates) <= 8 else f"{len(updates)} updates"})
            print(f"  {name:24s} done; updates {len(updates)}")

    table = pd.DataFrame(results)
    print(f"\n=== results; control (no drift) F1 in drift range = {ctl['F1']:.4f} ===")
    print(table.round(4).to_string(index=False))

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for a, kind in zip(ax, ["sudden", "gradual"]):
        for (k, name), w in wf1s.items():
            if k == kind:
                a.plot(pd.Series(w).rolling(20, min_periods=1).mean(), label=name, lw=1.1)
        a.axvline(W0, color="k", ls=":")
        a.set_ylabel(f"{kind}: window F1 (rolling 20)")
        a.legend(fontsize=8)
    ax[-1].set_xlabel(f"stream window (dotted = drift starts at {W0})")
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "step8_drift.png"), dpi=110)
    pd.to_pickle({"table": table, "detection": pd.DataFrame(detect), "mag": mag, "control": ctl,
                  "window_f1": wf1s, "updates": upd}, os.path.join(FOLDER, "step8_results.pkl"))
    print("\nsaved step8_results.pkl and step8_drift.png")
    plt.show()


if __name__ == "__main__":
    main()
