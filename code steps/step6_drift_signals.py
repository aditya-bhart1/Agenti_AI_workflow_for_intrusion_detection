"""
Step 6: AISAF drift signals on the per-window results from Step 5. Signals only: nothing is adapted yet.

Per window t (512 flows each), following AISAF Sec. III-C:
  delta_t = ||z_t - z_{t-1}|| / ||z_{t-1}||            Eq. 5  latent-space deviation
  dc_t    = c_t - cbar_{t-1}                            Eq. 6  confidence vs its moving average (alpha = 0.2)
  S_t     = lam*delta_t + (1-lam)*(-dc_t)               Eq. 7  fused score (lam = 0.5)
  T_t     = mean(S) + k*std(S) over the last 512 windows Eq. 8  adaptive threshold (k = 1.5)
Per-signal flags from Sec. III-C-6:  delta_t > mean(delta)+1.5*std(delta)   and   dc_t < -0.04
A trigger needs the flag to hold for 3 consecutive windows, with at most one trigger per 20 windows.

Two trigger rules are reported because the text is ambiguous about which one is meant:
  "fused": S_t > T_t for 3 windows                (Eq. 7-8)
  "both" : latent flag AND confidence flag for 3 windows ("all drift conditions")
We also compute delta_ref (distance to the average validation embedding). That one is OUR addition,
used for diagnosis only. It is not part of any trigger.

The validation windows come first and only build up history (no triggers are counted there).

Reads : step5_windows.pkl     Writes: step6_drift.pkl, step6_signals.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- settings (values from the paper unless marked) ---------------------
FOLDER = os.environ.get("UNSW_FOLDER", r"C:\Users\Lenovo\Documents\pbl\dataset one")
LAM = 0.5
K = 1.5
ALPHA = 0.2
DC_THRESHOLD = -0.04
PERSIST = 3
COOLDOWN = 20
HISTORY = 512          # windows used for the running mean/std (Eq. 8)
MIN_HISTORY = 20       # OUR choice: no thresholds until this many windows exist
# -------------------------------------------------------------------------

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def _norm(a):
    return np.linalg.norm(a, axis=1)


def trigger_rule(flag, start_at, persist=PERSIST, cooldown=COOLDOWN):
    """flag held for `persist` consecutive windows; at most one trigger per `cooldown` windows;
    nothing before window index `start_at`."""
    flag = np.asarray(flag, dtype=bool)
    held = pd.Series(flag).rolling(persist).sum().fillna(0).values == persist
    trig = np.zeros(len(flag), dtype=bool)
    last = -10**9
    for t in range(start_at, len(flag)):
        if held[t] and t - last >= cooldown:
            trig[t] = True
            last = t
    return trig


def run_detector(emb, conf, n_ref):
    """emb: (n_windows, d) mean embedding per window; conf: (n_windows,) mean confidence.
    The first n_ref windows are the reference (validation) period."""
    n = len(conf)
    delta = np.full(n, np.nan)
    delta[1:] = _norm(emb[1:] - emb[:-1]) / np.maximum(_norm(emb[:-1]), 1e-12)          # Eq. 5

    cbar = np.zeros(n)
    dc = np.full(n, np.nan)
    cbar[0] = conf[0]
    for t in range(1, n):
        dc[t] = conf[t] - cbar[t - 1]                                                   # Eq. 6
        cbar[t] = ALPHA * conf[t] + (1 - ALPHA) * cbar[t - 1]

    S = LAM * delta + (1 - LAM) * (-dc)                                                 # Eq. 7

    def running(x):          # statistics of the PREVIOUS windows only (the current one is excluded)
        r = pd.Series(x).rolling(HISTORY, min_periods=MIN_HISTORY)
        return r.mean().shift(1).values, r.std().shift(1).values

    mu_s, sd_s = running(S)
    mu_d, sd_d = running(delta)
    T = mu_s + K * sd_s                                                                 # Eq. 8
    delta_thr = mu_d + K * sd_d

    latent_flag = delta > delta_thr
    conf_flag = dc < DC_THRESHOLD
    fused_flag = S > T
    ref = emb[:n_ref].mean(axis=0)
    delta_ref = _norm(emb - ref) / max(np.linalg.norm(ref), 1e-12)                       # our addition

    return pd.DataFrame({
        "delta": delta, "delta_thr": delta_thr, "delta_ref": delta_ref,
        "conf": conf, "cbar": cbar, "dc": dc,
        "S": S, "T": T,
        "latent_flag": latent_flag, "conf_flag": conf_flag, "fused_flag": fused_flag,
        "trigger_fused": trigger_rule(fused_flag, n_ref),
        "trigger_both": trigger_rule(latent_flag & conf_flag, n_ref),
    })


def plot_signals(df, n_ref, path):
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    x = np.arange(len(df))
    ax[0].plot(x, df["delta"], lw=0.8, label="delta (Eq. 5)")
    ax[0].plot(x, df["delta_thr"], "r--", lw=0.8, label="mean + 1.5 std")
    ax[0].set_ylabel("latent deviation")
    ax[1].plot(x, df["dc"], lw=0.8, label="confidence change (Eq. 6)")
    ax[1].axhline(DC_THRESHOLD, color="r", ls="--", lw=0.8, label=f"{DC_THRESHOLD}")
    ax[1].set_ylabel("confidence change")
    ax[2].plot(x, df["S"], lw=0.8, label="fused score S (Eq. 7)")
    ax[2].plot(x, df["T"], "r--", lw=0.8, label="adaptive threshold T (Eq. 8)")
    ax[2].set_ylabel("fused score")
    ax[3].plot(x, df["delta_ref"], color="tab:green", lw=0.8, label="distance to validation average (our addition)")
    ax[3].set_ylabel("delta_ref")
    if "f1" in df:
        ax[4].plot(x, df["f1"], ".", ms=3, alpha=0.5, label="window F1")
        ax[4].plot(x, df["f1"].rolling(20, min_periods=1).mean(), lw=1.2, label="rolling mean (20)")
    ax[4].set_ylabel("window F1")
    ax[4].set_xlabel("window index (validation first, then stream)")
    for a in ax:
        a.axvline(n_ref, color="k", ls=":", lw=1)
        for t in df.index[df["trigger_fused"]]:
            a.axvline(t, color="red", alpha=0.5, lw=1)
        for t in df.index[df["trigger_both"]]:
            a.axvline(t, color="orange", alpha=0.5, lw=1)
        a.legend(loc="upper right", fontsize=7)
    ax[0].set_title("dotted line = stream starts; red line = trigger (fused rule); orange line = trigger (both-signals rule)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    return fig


def main():
    w = pd.read_pickle(os.path.join(FOLDER, "step5_windows.pkl"))
    val, stream = w["val"], w["stream"]
    n_ref = len(val["table"])
    emb = np.vstack([val["embeddings"], stream["embeddings"]])
    tbl = pd.concat([val["table"], stream["table"]], ignore_index=True)
    sig = run_detector(emb, tbl["mean_conf"].values, n_ref)
    out = pd.concat([tbl[["start", "attacks", "attack_share", "f1", "false_alarm"]], sig], axis=1)
    out.insert(0, "split", ["val"] * n_ref + ["stream"] * (len(out) - n_ref))

    s = out[out["split"] == "stream"]
    print(f"{n_ref} reference (validation) windows, {len(s)} stream windows\n")
    print("settings: lam={}, k={}, alpha={}, dc threshold={}, persistence={}, cooldown={}, history={}".format(
        LAM, K, ALPHA, DC_THRESHOLD, PERSIST, COOLDOWN, HISTORY))
    print("\nstream windows flagged:")
    for c in ["latent_flag", "conf_flag", "fused_flag"]:
        print(f"  {c:12s} {int(s[c].sum()):>5d}  ({s[c].mean():.1%})")
    print(f"triggers on the stream: fused rule = {int(s['trigger_fused'].sum())}, "
          f"both-signals rule = {int(s['trigger_both'].sum())}")
    cols = ["split", "start", "attack_share", "f1", "delta", "dc", "S", "T"]
    for name in ["trigger_fused", "trigger_both"]:
        hit = out[out[name]]
        if len(hit):
            print(f"\n{name} windows:")
            print(hit[cols].round({c: 4 for c in ["attack_share", "f1", "delta", "dc", "S", "T"]}).to_string())
    print("\nstream signal summary:")
    print(s[["delta", "dc", "S", "delta_ref"]].describe().round(4))
    print("\nvalidation (reference) signal summary, for comparison:")
    print(out[out["split"] == "val"][["delta", "dc", "S", "delta_ref"]].describe().round(4))
    rnd = {c: 4 for c in ["attack_share", "f1", "delta", "dc", "S", "T"]}
    for t in out.index[out["trigger_fused"]]:
        print(f"\ncontext: windows {t - 2} to {t + 1} (the trigger is window {t}):")
        print(out.loc[t - 2:t + 1, ["start", "attack_share", "f1", "delta", "dc", "S", "T"]].round(rnd).to_string())

    pd.to_pickle({"signals": out, "n_ref": n_ref}, os.path.join(FOLDER, "step6_drift.pkl"))
    plot_signals(out, n_ref, os.path.join(FOLDER, "step6_signals.png"))
    print("\nsaved step6_drift.pkl and step6_signals.png")
    plt.show()
    return out


if __name__ == "__main__":
    out = main()