"""
Step 7: selective update of the model while the stream runs (AISAF Sec. III-D idea, simplified).

The stream is replayed window by window (512 flows). For each window the CURRENT model predicts first
(test-then-train), then the controller may decide to update the model:
  - labels for window t only become available at window t + LABEL_DELAY      (our design choice)
  - an update fine-tunes everything EXCEPT the CNN (LSTM, Transformer, fuse, head) on the most recent
    N_RECENT labelled windows plus N_REPLAY random TRAIN flows (replay, to limit forgetting)
  - plain fine-tuning with a small learning rate; the paper's meta-update (Eq. 10) is NOT implemented.

Runs compared (each starts from the same Step 4 weights):
  none      : never update                              (baseline)
  detector  : our label-free trigger from Step 6c       (-conf of flows called "attack", smoothed, val mean+K*std,
              3 windows in a row, cooldown 20)
  oracle    : update right after each bad stretch begins (stretches found from the 'none' run: it knows when
              the model is bad, but still only trains on labels that exist by then)
  periodic  : blind update every PERIOD windows          (does a smart trigger beat a timer?)

Reports: overall and in-stretch F1, F1 outside stretches (forgetting check), Fuzzers/Exploits recall in
stretches, number and position of updates, plus a plot. ONE seed only: treat small gaps as noise.

Reads : step2_data.pkl, step4_model.pt, step4_results.pkl      Writes: step7_results.pkl, step7_update.png
Needs : torch, matplotlib      (GPU recommended; about 2-4 minutes in total)
"""
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---- settings -----------------------------------------------------------
FOLDER = os.environ.get("UNSW_FOLDER", r"C:\Users\Lenovo\Documents\pbl\dataset one")
WINDOW = 512
LABEL_DELAY = int(os.environ.get("LABEL_DELAY", 3))      # windows until a window's labels are known
N_RECENT = int(os.environ.get("N_RECENT", 20))           # labelled windows used per update
N_REPLAY = int(os.environ.get("N_REPLAY", 10000))        # random train flows mixed into each update
UPDATE_EPOCHS = int(os.environ.get("UPDATE_EPOCHS", 2))
UPDATE_LR = float(os.environ.get("UPDATE_LR", 1e-4))
BATCH = 128
PERIOD = int(os.environ.get("PERIOD", 100))
SMOOTH, K, PERSIST, COOLDOWN = 10, 2.0, 3, 20            # detector (same as Step 6c)
ORACLE_OFFSET = int(os.environ.get("ORACLE_OFFSET", LABEL_DELAY + 3))
BAD_F1, F1_ROLL, MIN_BAD_LEN = 0.90, 20, 10             # how bad stretches are defined (same as Step 6c)
SEED = int(os.environ.get("SEED", 0))
# -------------------------------------------------------------------------

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


# ---- model code: copy of step4 (keep in sync) --------------------------------
def make_batch(xs, rows, T):
    offsets = torch.arange(T - 1, -1, -1, device=rows.device)
    ids = (rows[:, None] - offsets[None, :]).clamp(min=0)
    return xs[ids]


class Backbone(nn.Module):
    def __init__(self, n_feat, T, d=64, heads=4, layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(8),
            nn.Flatten(), nn.Linear(32 * 8, d), nn.ReLU())
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.fuse = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(0.1))
        self.head = nn.Linear(d, 1)

    def forward(self, x, return_embedding=False):
        B, T, F = x.shape
        e = self.cnn(x.reshape(B * T, 1, F)).reshape(B, T, -1)
        h, (h_last, _) = self.lstm(e)
        z = self.transformer(h + self.pos[:, :T])
        fused = self.fuse(torch.cat([e.mean(1), h_last[-1], z.mean(1)], dim=1))
        logit = self.head(fused).squeeze(1)
        return (logit, fused) if return_embedding else logit


# ---- numpy helpers (no torch) ------------------------------------------------
def window_conf_attack(prob_2d):
    """Detector signal per window: minus the mean confidence of flows called 'attack'."""
    pred = prob_2d > 0.5
    conf = np.maximum(prob_2d, 1 - prob_2d)
    n = pred.sum(1)
    return -np.where(n > 0, (conf * pred).sum(1) / np.maximum(n, 1), np.nan)


class DetectorState:
    """Online version of the Step 6c detector: smoothed signal > threshold for PERSIST windows, cooldown."""
    def __init__(self, history, thr):
        self.hist = list(history[-(SMOOTH - 1):])
        self.thr, self.run, self.last = thr, 0, -10**9

    def step(self, t, sig):
        if np.isnan(sig):                       # no flow called attack: reuse the last value
            sig = self.hist[-1] if self.hist else -1.0
        self.hist.append(sig)
        self.hist = self.hist[-SMOOTH:]
        flag = np.mean(self.hist) > self.thr
        self.run = self.run + 1 if flag else 0
        if self.run >= PERSIST and t - self.last >= COOLDOWN:
            self.last = t
            return True
        return False


def f1_from(tp, fp, fn):
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else np.nan


def bad_stretches(f1_windows):
    c = pd.Series(f1_windows).rolling(F1_ROLL, center=True, min_periods=5).mean().values
    bad = c < BAD_F1
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    segs = [(int(a), int(b) - 1) for a, b in zip(edges[0::2], edges[1::2]) if b - a >= MIN_BAD_LEN]
    mask = np.zeros(len(f1_windows), bool)
    for a, b in segs:
        mask[a:b + 1] = True
    return segs, mask


def window_counts(y2d, pred2d):
    tp = (y2d & pred2d).sum(1)
    fp = (~y2d & pred2d).sum(1)
    fn = (y2d & ~pred2d).sum(1)
    return tp, fp, fn


def summarise(name, y2d, pred2d, cats2d, mask, segs, updates):
    tp, fp, fn = window_counts(y2d, pred2d)
    wf1 = np.array([f1_from(a, b, c) for a, b, c in zip(tp, fp, fn)])

    def pooled(sel):
        a, b, c = tp[sel].sum(), fp[sel].sum(), fn[sel].sum()
        negs = (~y2d[sel]).sum()
        return f1_from(a, b, c), a / max(a + c, 1), b / max(negs, 1)

    allw = np.ones(len(tp), bool)
    f_all, r_all, fa_all = pooled(allw)
    f_in, r_in, fa_in = pooled(mask)
    f_out, _, _ = pooled(~mask)
    row = {"run": name, "updates": len(updates), "F1 all": f_all, "recall all": r_all, "false alarm all": fa_all,
           "F1 in stretches": f_in, "recall in stretches": r_in, "false alarm in stretches": fa_in,
           "F1 outside": f_out}
    for k, (a, b) in enumerate(segs, 1):
        sel = np.zeros(len(tp), bool)
        sel[a:b + 1] = True
        row[f"F1 stretch {k}"] = pooled(sel)[0]
    for cat in ["Fuzzers", "Exploits"]:
        sel = mask[:, None] & y2d & (cats2d == cat)
        row[f"{cat} recall (stretches)"] = pred2d[sel].mean() if sel.any() else np.nan
    return row, wf1


# ---- torch parts -------------------------------------------------------------
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_rows(model, xs, rows, T):
    model.eval()
    return torch.sigmoid(model(make_batch(xs, rows, T))).cpu().numpy()


def fine_tune(model, Xs, Ys, Xtr, Ytr, t, T, gen):
    """Update on the most recent labelled windows (those <= t - LABEL_DELAY) plus replay from train."""
    last = t - LABEL_DELAY
    if last < 0:
        return False
    first = max(0, last - N_RECENT + 1)
    stream_rows = torch.arange(first * WINDOW, (last + 1) * WINDOW, device=dev)
    replay_rows = torch.randint(0, Xtr.shape[0], (N_REPLAY,), device=dev, generator=gen)
    for p in model.cnn.parameters():
        p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=UPDATE_LR)
    loss_fn = nn.BCEWithLogitsLoss()
    n_s, n_r = len(stream_rows), len(replay_rows)
    model.train()
    for _ in range(UPDATE_EPOCHS):
        perm = torch.randperm(n_s + n_r, device=dev, generator=gen)
        for s in range(0, n_s + n_r, BATCH):
            idx = perm[s:s + BATCH]
            from_stream = idx < n_s
            xs_parts, ys_parts = [], []
            if from_stream.any():
                r = stream_rows[idx[from_stream]]
                xs_parts.append(make_batch(Xs, r, T))
                ys_parts.append(Ys[r])
            if (~from_stream).any():
                r = replay_rows[idx[~from_stream] - n_s]
                xs_parts.append(make_batch(Xtr, r, T))
                ys_parts.append(Ytr[r])
            loss = loss_fn(model(torch.cat(xs_parts)), torch.cat(ys_parts))
            opt.zero_grad()
            loss.backward()
            opt.step()
    for p in model.cnn.parameters():
        p.requires_grad = True
    return True


def run_stream(mode, base_state, ctx, oracle_at=None):
    Xs, Ys, Xtr, Ytr, T, n_w, thr, val_sig_tail = (ctx[k] for k in
        ["Xs", "Ys", "Xtr", "Ytr", "T", "n_w", "thr", "val_sig_tail"])
    model = Backbone(ctx["n_feat"], T, ctx["d"]).to(dev)
    model.load_state_dict(copy.deepcopy(base_state))
    gen = torch.Generator(device=dev)
    gen.manual_seed(SEED)
    det = DetectorState(val_sig_tail, thr)
    probs = np.zeros((n_w, WINDOW), dtype="float32")
    updates = []
    for t in range(n_w):
        rows = torch.arange(t * WINDOW, (t + 1) * WINDOW, device=dev)
        probs[t] = predict_rows(model, Xs, rows, T)
        sig = window_conf_attack(probs[t:t + 1])[0]
        fire = det.step(t, sig)
        do_update = {"none": False,
                     "detector": fire,
                     "oracle": oracle_at is not None and t in oracle_at,
                     "periodic": (t + 1) % PERIOD == 0}[mode]
        if do_update and fine_tune(model, Xs, Ys, Xtr, Ytr, t, T, gen):
            updates.append(t)
    return probs, updates


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    d2 = pd.read_pickle(os.path.join(FOLDER, "step2_data.pkl"))
    d4 = pd.read_pickle(os.path.join(FOLDER, "step4_results.pkl"))
    ckpt = torch.load(os.path.join(FOLDER, "step4_model.pt"), map_location=dev)
    T, n_feat, d = ckpt["T"], ckpt["n_feat"], ckpt["d"]
    print(f"device {dev}; model T={T}; label delay {LABEL_DELAY}, recent {N_RECENT} windows, replay {N_REPLAY}, "
          f"{UPDATE_EPOCHS} epochs, lr {UPDATE_LR}")

    n_w = len(d2["y"]["stream"]) // WINDOW
    m = n_w * WINDOW
    y2d = d2["y"]["stream"][:m].astype(bool).reshape(n_w, WINDOW)
    cats2d = np.asarray(d2["meta"]["stream"]["attack_cat"].astype(str).to_numpy(dtype=object)[:m],
                        dtype=object).reshape(n_w, WINDOW)

    # detector threshold from VALIDATION with the original model's outputs (as in Step 6c)
    nv = len(d2["y"]["val"]) // WINDOW
    val_sig = window_conf_attack(d4["prob"]["val"][:nv * WINDOW].reshape(nv, WINDOW))
    val_sig = pd.Series(val_sig).ffill().fillna(-1.0).values
    sm = pd.Series(val_sig).rolling(SMOOTH, min_periods=1).mean().values
    thr = float(sm.mean() + K * sm.std())
    print(f"detector threshold {thr:.4f} (val mean {sm.mean():.4f}, std {sm.std():.4f})")

    ctx = {"Xs": torch.tensor(d2["X"]["stream"], device=dev),
           "Ys": torch.tensor(d2["y"]["stream"], dtype=torch.float32, device=dev),
           "Xtr": torch.tensor(d2["X"]["train"], device=dev),
           "Ytr": torch.tensor(d2["y"]["train"], dtype=torch.float32, device=dev),
           "T": T, "n_w": n_w, "n_feat": n_feat, "d": d, "thr": thr, "val_sig_tail": list(val_sig[-(SMOOTH - 1):])}
    base_state = ckpt["state"]

    results, wf1s, upd = [], {}, {}
    t0 = time.time()
    probs, updates = run_stream("none", base_state, ctx)
    pred = probs > 0.5
    tp, fp, fn = window_counts(y2d, pred)
    wf1_none = np.array([f1_from(a, b, c) for a, b, c in zip(tp, fp, fn)])
    segs, mask = bad_stretches(wf1_none)
    print(f"bad stretches (from the no-update run): {segs}   [{time.time() - t0:.0f}s]")
    row, wf1s["none"] = summarise("none", y2d, pred, cats2d, mask, segs, updates)
    results.append(row)
    upd["none"] = updates

    oracle_at = {min(a + ORACLE_OFFSET, n_w - 1) for a, _ in segs}
    for mode in ["detector", "oracle", "periodic"]:
        t0 = time.time()
        probs, updates = run_stream(mode, base_state, ctx, oracle_at)
        row, wf1s[mode] = summarise(mode, y2d, probs > 0.5, cats2d, mask, segs, updates)
        results.append(row)
        upd[mode] = updates
        print(f"{mode:9s} done in {time.time() - t0:.0f}s; updates at windows {updates}")

    table = pd.DataFrame(results)
    print("\n=== results (updates are applied AFTER the window that triggers them; test-then-train) ===")
    print(table.round(4).T.to_string(header=False))
    base = table.iloc[0]
    print("\ndifference vs no-update (F1 points x100):")
    diff = table.set_index("run")[["F1 all", "F1 in stretches", "F1 outside"] +
                                  [c for c in table.columns if c.startswith("F1 stretch")]]
    print(((diff - diff.loc["none"]) * 100).round(2).to_string())

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, w in wf1s.items():
        ax[0].plot(pd.Series(w).rolling(F1_ROLL, min_periods=1).mean(), label=name, lw=1.1)
    for a, b in segs:
        for x in ax:
            x.axvspan(a, b, color="grey", alpha=0.2)
    ax[0].set_ylabel(f"window F1 (rolling {F1_ROLL})")
    ax[0].legend()
    for i, (name, u) in enumerate(upd.items()):
        ax[1].vlines(u, i, i + 0.8)
    ax[1].set_yticks(np.arange(len(upd)) + 0.4)
    ax[1].set_yticklabels(list(upd))
    ax[1].set_xlabel("stream window index (grey = bad stretch, black marks = updates)")
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "step7_update.png"), dpi=110)
    pd.to_pickle({"table": table, "window_f1": wf1s, "updates": upd, "segments": segs},
                 os.path.join(FOLDER, "step7_results.pkl"))
    print("\nsaved step7_results.pkl and step7_update.png")
    plt.show()


if __name__ == "__main__":
    main()