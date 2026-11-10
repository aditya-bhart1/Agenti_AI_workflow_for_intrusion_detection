"""
Step 4: sequence windows + CNN -> LSTM -> Transformer backbone (AISAF Sec. III-B).

What it does
  1. Builds a window of T consecutive flows for every row (window i = rows i-T+1 .. i of
     the same split; the first rows of a split repeat row 0). The label is the LAST flow's.
  2. Trains the backbone for a few epochs for each T in T_CANDIDATES and compares them
     on VALIDATION F1 (the stream is never used to choose anything).
  3. Retrains the best T for FINAL_EPOCHS, keeps the epoch with the best validation F1,
     then reports on val / stream / stream-novel-rows next to the Step 3 baselines.

Paper settings used: Adam lr 1e-3 with cosine decay, batch 128, 30 epochs, concatenation of
CNN / LSTM / Transformer embeddings. Layer sizes are OUR choice (the paper text gives none).

Needs: pip install torch scikit-learn
"""
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---- settings -----------------------------------------------------------
FOLDER = os.environ.get("UNSW_FOLDER", r"C:\Users\Lenovo\Documents\pbl\dataset one")
T_CANDIDATES = [int(t) for t in os.environ.get("T_CANDIDATES", "1,4,16,64").split(",")]
SEARCH_EPOCHS = int(os.environ.get("SEARCH_EPOCHS", 3))
FINAL_EPOCHS = int(os.environ.get("FINAL_EPOCHS", 30))
T_FINAL = int(os.environ["T_FINAL"]) if "T_FINAL" in os.environ else None  # None = best on val
BATCH = 128
LR = 1e-3
D = 64            # embedding width (our choice)
SEED = 0
# -------------------------------------------------------------------------

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev)

data = pd.read_pickle(os.path.join(FOLDER, "step2_data.pkl"))
X = {k: torch.tensor(v, device=dev) for k, v in data["X"].items()}
Y = {k: torch.tensor(v, dtype=torch.float32, device=dev) for k, v in data["y"].items()}
y_np = data["y"]
meta = data["meta"]
N_FEAT = X["train"].shape[1]


# --------------------------------------------------------------------- windows
def make_batch(xs, rows, T):
    """rows: tensor of row ids -> (B, T, n_features) windows ending at those rows."""
    offsets = torch.arange(T - 1, -1, -1, device=rows.device)
    ids = (rows[:, None] - offsets[None, :]).clamp(min=0)
    return xs[ids]


# ----------------------------------------------------------------------- model
class Backbone(nn.Module):
    def __init__(self, n_feat, T, d=D, heads=4, layers=2):
        super().__init__()
        # CNN over the feature axis of ONE flow (as the paper describes: across feature dimensions)
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
        e = self.cnn(x.reshape(B * T, 1, F)).reshape(B, T, -1)   # CNN embedding per flow
        h, (h_last, _) = self.lstm(e)                              # LSTM over the window
        z = self.transformer(h + self.pos[:, :T])                  # Transformer over LSTM states
        fused = self.fuse(torch.cat([e.mean(1), h_last[-1], z.mean(1)], dim=1))
        logit = self.head(fused).squeeze(1)
        return (logit, fused) if return_embedding else logit      # fused = embedding for Step 6


@torch.no_grad()
def predict_proba(model, split, T, bs=4096):
    model.eval()
    n = X[split].shape[0]
    out = []
    for s in range(0, n, bs):
        rows = torch.arange(s, min(s + bs, n), device=dev)
        out.append(torch.sigmoid(model(make_batch(X[split], rows, T))))
    return torch.cat(out).cpu().numpy()


def train(T, epochs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = Backbone(N_FEAT, T).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = X["train"].shape[0]
    steps = epochs * ((n + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss_fn = nn.BCEWithLogitsLoss()
    best = {"f1": -1.0, "state": None, "epoch": 0}
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, device=dev)
        total = torch.zeros((), device=dev)
        for s in range(0, n, BATCH):
            rows = perm[s:s + BATCH]
            loss = loss_fn(model(make_batch(X["train"], rows, T)), Y["train"][rows])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.detach() * len(rows)
        f1 = f1_score(y_np["val"], predict_proba(model, "val", T) > 0.5, zero_division=0)
        print(f"  T={T:<3d} epoch {ep:>2d}/{epochs}  train loss {total.item() / n:.4f}  "
              f"val F1 {f1:.4f}  ({time.time() - t0:.0f}s)")
        if f1 > best["f1"]:
            best = {"f1": f1, "state": copy.deepcopy(model.state_dict()), "epoch": ep}
    model.load_state_dict(best["state"])
    return model, best


# --------------------------------------------------------------------- scoring
def scores(y_true, y_pred):
    return {"rows": len(y_true),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred)}


def per_type_detection(meta_df, pred):
    flagged = pd.Series(pred, index=meta_df.index)
    out = flagged.groupby(meta_df["attack_cat"]).agg(rows="size", flagged="mean")
    out.index = [("false alarm rate" if c == "Normal" else f"{c} recall") for c in out.index]
    return out


def hourly_f1(meta_df, y_true, pred):
    df = pd.DataFrame({"hour": meta_df["Stime"].dt.floor("h"), "y": y_true, "p": pred})
    return df.groupby("hour").apply(
        lambda g: pd.Series({"rows": len(g), "attacks": int(g["y"].sum()),
                             "F1": f1_score(g["y"], g["p"], zero_division=0)}),
        include_groups=False)


# ------------------------------------------------------------------------ main
print("\n=== 1. choose T on VALIDATION ===")
search = []
for T in T_CANDIDATES:
    t0 = time.time()
    _, best = train(T, SEARCH_EPOCHS)
    search.append({"T": T, "best val F1": best["f1"], "seconds": time.time() - t0})
search = pd.DataFrame(search)
print(search.round(4).to_string(index=False))
T_best = int(search.loc[search["best val F1"].idxmax(), "T"]) if T_FINAL is None else T_FINAL
print(f"\nusing T = {T_best}")

print(f"\n=== 2. final training: T={T_best}, {FINAL_EPOCHS} epochs ===")
model, best = train(T_best, FINAL_EPOCHS)
print(f"kept epoch {best['epoch']} (val F1 {best['f1']:.4f})")

prob = {s: predict_proba(model, s, T_best) for s in ["val", "stream"]}
pred = {s: (prob[s] > 0.5).astype(int) for s in prob}

step3_path = os.path.join(FOLDER, "step3_baselines.pkl")
step3 = pd.read_pickle(step3_path) if os.path.exists(step3_path) else None
novel = step3["novel"]["stream"] if step3 else None

rows = [{"model": f"CNN-LSTM-Transformer (T={T_best})", "eval on": "val", **scores(y_np["val"], pred["val"])},
        {"model": f"CNN-LSTM-Transformer (T={T_best})", "eval on": "stream", **scores(y_np["stream"], pred["stream"])}]
if novel is not None:
    rows.append({"model": f"CNN-LSTM-Transformer (T={T_best})", "eval on": "stream (novel rows only)",
                 **scores(y_np["stream"][novel], pred["stream"][novel])})
else:
    print("note: step3_baselines.pkl not found, skipping the novel-rows line")
neural = pd.DataFrame(rows)

print("\n=== summary (neural model next to the Step 3 baselines) ===")
table = pd.concat([step3["summary"], neural]) if step3 else neural
print(table.round(4).to_string(index=False))

print("\ndetection by attack type on the stream:")
print(per_type_detection(meta["stream"], pred["stream"]).round(4))
print("\nstream F1 per hour:")
print(hourly_f1(meta["stream"], y_np["stream"], pred["stream"]).round(4))

torch.save({"state": model.state_dict(), "T": T_best, "n_feat": N_FEAT, "d": D,
            "feature_names": data["feature_names"]}, os.path.join(FOLDER, "step4_model.pt"))
pd.to_pickle({"search": search, "summary": neural, "prob": prob, "T": T_best},
             os.path.join(FOLDER, "step4_results.pkl"))
print("\nsaved step4_model.pt and step4_results.pkl")