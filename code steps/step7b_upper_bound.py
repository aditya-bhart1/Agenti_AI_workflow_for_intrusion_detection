"""
Step 7b: can ANY update fix the bad stretches? (upper-bound test, uses labels on purpose)

Takes the Step 4 model and fine-tunes it on data from the bad stretches, then measures F1 on stretches:
  same-data  : train on stretch k, test on stretch k          (ceiling; memorisation allowed)
  half-half  : train on the first half of stretch k, test on its second half
  cross      : train on stretch 1, test on stretch 2 (and the reverse)
Two update types: "non-CNN" (CNN frozen, as in Step 7) and "all layers".
Also reports F1 on all windows OUTSIDE the stretches before/after (forgetting check).

How to read: if even same-data barely beats the original model, these flows cannot be separated with the
current features/model, and no update strategy will fix the dips. If same-data jumps but half-half and
cross do not, the stretches are not repeatable patterns and updates cannot generalise to them.

Reads : step2_data.pkl, step4_model.pt, step4_results.pkl, step7_selective_update.py (same folder)
Writes: step7b_results.pkl            Takes about 2-5 minutes on the GPU.
"""
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import step7_selective_update as s7      # reuses Backbone, make_batch, bad_stretches (keeps the code in sync)

FOLDER, WINDOW, dev = s7.FOLDER, s7.WINDOW, s7.dev
LR = float(os.environ.get("UB_LR", 1e-3))
EPOCHS = int(os.environ.get("UB_EPOCHS", 5))
N_REPLAY = int(os.environ.get("UB_REPLAY", 5000))
BATCH = 128
SEED = int(os.environ.get("SEED", 0))
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def rows_of(wins):
    return np.concatenate([np.arange(w * WINDOW, (w + 1) * WINDOW) for w in wins])


@torch.no_grad()
def predict_many(model, xs, rows_np, T, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(rows_np), bs):
        r = torch.as_tensor(rows_np[i:i + bs], device=dev)
        out.append(torch.sigmoid(model(s7.make_batch(xs, r, T))).cpu().numpy())
    return np.concatenate(out)


def score(model, ctx, wins):
    rows = rows_of(wins)
    pred = predict_many(model, ctx["Xs"], rows, ctx["T"]) > 0.5
    y, cat = ctx["y"][rows], ctx["cat"][rows]
    tp, fp, fn = (y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum()
    fz = y & (cat == "Fuzzers")
    return {"F1": s7.f1_from(tp, fp, fn), "recall": tp / max(tp + fn, 1),
            "false alarm": fp / max((~y).sum(), 1), "Fuzzers recall": pred[fz].mean() if fz.any() else np.nan}


def fine_tune_on(model, ctx, train_wins, freeze_cnn, gen):
    stream_rows = torch.as_tensor(rows_of(train_wins), device=dev)
    replay = torch.randint(0, ctx["Xtr"].shape[0], (N_REPLAY,), device=dev, generator=gen)
    for p in model.cnn.parameters():
        p.requires_grad = not freeze_cnn
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    loss_fn, T = nn.BCEWithLogitsLoss(), ctx["T"]
    n_s, n_r = len(stream_rows), len(replay)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n_s + n_r, device=dev, generator=gen)
        for i in range(0, n_s + n_r, BATCH):
            idx = perm[i:i + BATCH]
            m = idx < n_s
            xs, ys = [], []
            if m.any():
                r = stream_rows[idx[m]]
                xs.append(s7.make_batch(ctx["Xs"], r, T)); ys.append(ctx["Ys"][r])
            if (~m).any():
                r = replay[idx[~m] - n_s]
                xs.append(s7.make_batch(ctx["Xtr"], r, T)); ys.append(ctx["Ytr"][r])
            loss = loss_fn(model(torch.cat(xs)), torch.cat(ys))
            opt.zero_grad(); loss.backward(); opt.step()
    for p in model.cnn.parameters():
        p.requires_grad = True
    return model


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    d2 = pd.read_pickle(os.path.join(FOLDER, "step2_data.pkl"))
    d4 = pd.read_pickle(os.path.join(FOLDER, "step4_results.pkl"))
    ck = torch.load(os.path.join(FOLDER, "step4_model.pt"), map_location=dev)
    T = ck["T"]
    n_w = len(d2["y"]["stream"]) // WINDOW
    m = n_w * WINDOW
    y = d2["y"]["stream"][:m].astype(bool)
    cat = np.asarray(d2["meta"]["stream"]["attack_cat"].astype(str).to_numpy(dtype=object)[:m], dtype=object)
    ctx = {"Xs": torch.tensor(d2["X"]["stream"], device=dev),
           "Ys": torch.tensor(d2["y"]["stream"], dtype=torch.float32, device=dev),
           "Xtr": torch.tensor(d2["X"]["train"], device=dev),
           "Ytr": torch.tensor(d2["y"]["train"], dtype=torch.float32, device=dev),
           "T": T, "y": y, "cat": cat}

    # stretches from the original model's per-window F1 (same definition as Step 7)
    pred = (d4["prob"]["stream"][:m] > 0.5).reshape(n_w, WINDOW)
    tp, fp, fn = s7.window_counts(y.reshape(n_w, WINDOW), pred)
    wf1 = np.array([s7.f1_from(a, b, c) for a, b, c in zip(tp, fp, fn)])
    segs, mask = s7.bad_stretches(wf1)
    print(f"device {dev}; stretches {segs}; lr {LR}, epochs {EPOCHS}, replay {N_REPLAY}")
    outside = [w for w in range(n_w) if not mask[w]]
    wins = [list(range(a, b + 1)) for a, b in segs]

    def fresh():
        mdl = s7.Backbone(ck["n_feat"], T, ck["d"]).to(dev)
        mdl.load_state_dict(copy.deepcopy(ck["state"]))
        return mdl

    base = fresh()
    base_out = score(base, ctx, outside)
    scen = []                                   # (name, train windows, test windows)
    for k, w in enumerate(wins, 1):
        h = len(w) // 2
        scen.append((f"same-data (stretch {k})", w, w))
        scen.append((f"half-half (stretch {k})", w[:h], w[h:]))
    if len(wins) >= 2:
        scen.append(("cross: train 1, test 2", wins[0], wins[1]))
        scen.append(("cross: train 2, test 1", wins[1], wins[0]))

    rows = []
    for freeze in [True, False]:
        label = "non-CNN" if freeze else "all layers"
        for name, tr, te in scen:
            before = score(base, ctx, te)
            gen = torch.Generator(device=dev); gen.manual_seed(SEED)
            mdl = fine_tune_on(fresh(), ctx, tr, freeze, gen)
            after, after_out = score(mdl, ctx, te), score(mdl, ctx, outside)
            rows.append({"update": label, "scenario": name, "test windows": len(te),
                         "F1 before": before["F1"], "F1 after": after["F1"], "gain": after["F1"] - before["F1"],
                         "recall before": before["recall"], "recall after": after["recall"],
                         "Fuzzers rec before": before["Fuzzers recall"], "Fuzzers rec after": after["Fuzzers recall"],
                         "F1 outside change": after_out["F1"] - base_out["F1"]})
            print(f"  {label:10s} {name:26s} F1 {before['F1']:.4f} -> {after['F1']:.4f}")
    table = pd.DataFrame(rows)
    print("\n=== upper-bound results (gain = F1 after - F1 before, on the test windows) ===")
    print(table.round(4).to_string(index=False))
    pd.to_pickle({"table": table, "segments": segs}, os.path.join(FOLDER, "step7b_results.pkl"))
    print("\nsaved step7b_results.pkl")


if __name__ == "__main__":
    main()
