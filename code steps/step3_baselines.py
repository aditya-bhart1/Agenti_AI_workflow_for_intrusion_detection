"""
Step 3: baselines (Logistic Regression, Random Forest) on the Step 2 split.

Trains on TRAIN, then reports on VAL and STREAM:
  - overall precision / recall / F1 / accuracy (binary: attack vs normal)
  - the same on "novel" stream rows (rows NOT identical to any train row)
  - recall per attack type, and false-alarm rate on normal traffic
  - F1 per hour of the stream (a first look at drift)
  - which features fall outside the train range most often (feature drift preview)

Needs: pip install scikit-learn
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

FOLDER = os.environ.get("UNSW_FOLDER", r"C:\Users\Lenovo\Documents\pbl\dataset one")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

data = pd.read_pickle(os.path.join(FOLDER, "step2_data.pkl"))
X, y, meta, names = data["X"], data["y"], data["meta"], data["feature_names"]


def row_hashes(arr):
    return pd.util.hash_pandas_object(pd.DataFrame(arr), index=False)


def novel_mask(split):
    """True for rows whose feature vector never appears in train (approximate:
    rare categories were merged into 'other' in Step 2)."""
    train_hashes = set(row_hashes(X["train"]).values)
    return ~row_hashes(X[split]).isin(train_hashes).values


def scores(y_true, y_pred):
    return {
        "rows": len(y_true),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def per_type_detection(meta_df, pred):
    """Share of each attack type flagged as attack; for Normal this is the false-alarm rate."""
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


def out_of_range_report(split, top=8):
    a = X[split]
    frac = (a > 1.0).mean(axis=0)
    s = pd.Series(frac, index=names).sort_values(ascending=False)
    print(f"\n{split}: features with the most values ABOVE the train max (scaled > 1):")
    print(s.head(top).round(4).to_string())


models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0),
}

novel = {"val": novel_mask("val"), "stream": novel_mask("stream")}
print(f"novel rows (not identical to a train row): val {novel['val'].mean():.1%}, "
      f"stream {novel['stream'].mean():.1%}")

results, preds = [], {}
for name, model in models.items():
    print(f"\n==== {name}: training on {len(y['train']):,} rows ...")
    model.fit(X["train"], y["train"])
    preds[name] = {s: model.predict(X[s]) for s in ["val", "stream"]}

    for label, split, mask in [("val", "val", None),
                               ("stream", "stream", None),
                               ("stream (novel rows only)", "stream", novel["stream"])]:
        yt, yp = y[split], preds[name][split]
        if mask is not None:
            yt, yp = yt[mask], yp[mask]
        results.append({"model": name, "eval on": label, **scores(yt, yp)})

    print(f"\n{name}: detection by attack type on the stream")
    print(per_type_detection(meta["stream"], preds[name]["stream"]).round(4))

    print(f"\n{name}: stream F1 per hour")
    print(hourly_f1(meta["stream"], y["stream"], preds[name]["stream"]).round(4))

print("\n==== summary ====")
summary = pd.DataFrame(results)
print(summary.round(4).to_string(index=False))

out_of_range_report("val")
out_of_range_report("stream")

pd.to_pickle({"summary": summary, "preds": preds, "novel": novel},
             os.path.join(FOLDER, "step3_baselines.pkl"))
print("\nsaved step3_baselines.pkl")
