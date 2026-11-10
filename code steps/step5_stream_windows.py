"""
Step 5: run the trained Step 4 model over the data in consecutive windows of 512 flows.

For every window it records: attack share, accuracy, F1, false-alarm rate, mean confidence,
and the average of the model's internal embedding (needed by the drift signals in Step 6).
It does this for the validation block (the "normal behaviour" reference) and the stream.

Reads : step2_data.pkl, step4_model.pt, (optional) step3_baselines.pkl for a Random Forest line
Writes: step5_windows.pkl, step5_stream.png
Needs : torch, matplotlib
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---- settings -----------------------------------------------------------
FOLDER = os.environ.get("UNSW_FOLDER", r"C:\Users\Lenovo\Documents\pbl\dataset one")
WINDOW = 512      # flows per window (AISAF uses 512-sample streaming windows)
ROLL = 20         # rolling-mean length, in windows, for the plot lines
# -------------------------------------------------------------------------

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev)


# ---- model code: an exact copy of step4_sequence_model.py (keep the two in sync) ----
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
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d,
                                           dropout=0.1, batch_first=True)
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


@torch.no_grad()
def run_model(model, x, T, bs=4096):
    """Per-flow probability of 'attack' and per-flow embedding."""
    model.eval()
    n = x.shape[0]
    probs, embs = [], []
    for s in range(0, n, bs):
        rows = torch.arange(s, min(s + bs, n), device=dev)
        logit, emb = model(make_batch(x, rows, T), return_embedding=True)
        probs.append(torch.sigmoid(logit))
        embs.append(emb)
    return torch.cat(probs).cpu().numpy(), torch.cat(embs).cpu().numpy()


# ---- window metrics (plain numpy) ----------------------------------------
def window_f1(y, pred, window=WINDOW):
    """F1 per consecutive window. NaN for a window with no attacks and no alarms."""
    n_w = len(y) // window
    m = n_w * window
    yy = np.asarray(y[:m]).astype(bool).reshape(n_w, window)
    pp = np.asarray(pred[:m]).astype(bool).reshape(n_w, window)
    tp = (yy & pp).sum(1)
    fp = (~yy & pp).sum(1)
    fn = (yy & ~pp).sum(1)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)


def window_table(y, prob, meta_df, emb, window=WINDOW, other_pred=None):
    """One row per full window of `window` flows (a partial last window is dropped)."""
    n_w = len(y) // window
    m = n_w * window
    yy = np.asarray(y[:m]).astype(bool).reshape(n_w, window)
    pp = (prob[:m] > 0.5).reshape(n_w, window)
    conf = np.maximum(prob[:m], 1 - prob[:m]).reshape(n_w, window)
    times = meta_df["Stime"].values[:m].reshape(n_w, window)
    fp = (~yy & pp).sum(1)
    tbl = pd.DataFrame({
        "window": np.arange(n_w),
        "start": times[:, 0],
        "end": times[:, -1],
        "attacks": yy.sum(1),
        "attack_share": yy.mean(1),
        "accuracy": (yy == pp).mean(1),
        "f1": window_f1(y, prob > 0.5, window),
        "false_alarm": fp / np.maximum((~yy).sum(1), 1),
        "mean_conf": conf.mean(1),
    })
    if other_pred is not None:
        tbl["f1_other"] = window_f1(y, other_pred, window)
    embs = emb[:m].reshape(n_w, window, -1).mean(1).astype("float32")   # mean embedding per window
    return tbl, embs


def plot_stream(tbl, path, roll=ROLL, other_label="Random Forest"):
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    t = pd.to_datetime(tbl["start"])
    ax[0].plot(t, tbl["f1"], ".", ms=3, alpha=0.4, color="tab:blue", label="CNN-LSTM-Transformer (per window)")
    ax[0].plot(t, tbl["f1"].rolling(roll, min_periods=1).mean(), color="tab:blue", label=f"rolling mean ({roll} windows)")
    if "f1_other" in tbl:
        ax[0].plot(t, tbl["f1_other"].rolling(roll, min_periods=1).mean(), "--", color="tab:orange", label=f"{other_label} (rolling mean)")
    ax[0].set_ylabel("F1 per window")
    ax[0].legend(loc="lower left", fontsize=8)
    ax[1].plot(t, tbl["mean_conf"], ".", ms=3, alpha=0.4, color="tab:green")
    ax[1].plot(t, tbl["mean_conf"].rolling(roll, min_periods=1).mean(), color="tab:green")
    ax[1].set_ylabel("mean confidence")
    ax[2].plot(t, tbl["attack_share"], color="tab:purple")
    ax[2].set_ylabel("attack share")
    ax[2].set_xlabel("time (UTC, 18 Feb 2015)")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    return fig


# ---- main ----------------------------------------------------------------
def main():
    data = pd.read_pickle(os.path.join(FOLDER, "step2_data.pkl"))
    ckpt = torch.load(os.path.join(FOLDER, "step4_model.pt"), map_location=dev)
    T, n_feat, d = ckpt["T"], ckpt["n_feat"], ckpt["d"]
    model = Backbone(n_feat, T, d).to(dev)
    model.load_state_dict(ckpt["state"])
    print(f"loaded Step 4 model (T={T})")

    step3_path = os.path.join(FOLDER, "step3_baselines.pkl")
    rf_pred = None
    if os.path.exists(step3_path):
        rf_pred = pd.read_pickle(step3_path)["preds"]["RandForest"]

    out = {"window": WINDOW, "T": T}
    for split in ["val", "stream"]:
        x = torch.tensor(data["X"][split], device=dev)
        prob, emb = run_model(model, x, T)
        other = rf_pred[split] if rf_pred is not None else None
        tbl, embs = window_table(data["y"][split], prob, data["meta"][split], emb, other_pred=other)
        out[split] = {"table": tbl, "embeddings": embs}
        print(f"\n=== {split}: {len(tbl)} windows of {WINDOW} flows "
              f"({len(prob) - len(tbl) * WINDOW} trailing flows dropped) ===")
        print(tbl[["accuracy", "f1", "false_alarm", "mean_conf", "attack_share"]].describe().round(4))

    s = out["stream"]["table"]
    print("\n10 lowest-F1 windows on the stream:")
    print(s.nsmallest(10, "f1")[["window", "start", "attacks", "attack_share", "f1", "false_alarm", "mean_conf"]].round(4).to_string(index=False))
    print("\ncorrelation across stream windows:")
    print(s[["f1", "false_alarm", "mean_conf", "attack_share"]].corr().round(3))
    if "f1_other" in s:
        print(f"\nstream windows where the neural model has higher F1 than Random Forest: "
              f"{(s['f1'] > s['f1_other']).mean():.1%}")

    pd.to_pickle(out, os.path.join(FOLDER, "step5_windows.pkl"))
    plot_stream(s, os.path.join(FOLDER, "step5_stream.png"))
    print("\nsaved step5_windows.pkl and step5_stream.png")
    plt.show()


if __name__ == "__main__":
    main()
