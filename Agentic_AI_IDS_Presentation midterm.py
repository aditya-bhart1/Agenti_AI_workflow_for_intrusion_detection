import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------
# PART 1 — Load, clean, train
# ---------------------------------------------------------------

PATH = "PortScan.pcap_ISCX.csv"

df = pd.read_csv(PATH)
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df["Label"] = df["Label"].map({"BENIGN": 0, "PortScan": 1})

X = df.drop(columns=["Label"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("=== Classifier performance (Random Forest) ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print()

# ---------------------------------------------------------------
# PART 2 — Wrap the classifier as an agent-callable tool
# ---------------------------------------------------------------

def classify_flow(flow_features: pd.DataFrame) -> dict:
    """
    Agent tool: classifies one or more network flows.
    Returns label + confidence, the shape an LLM agent would call
    via function-calling.
    """
    pred = clf.predict(flow_features)[0]
    proba = clf.predict_proba(flow_features)[0]
    confidence = float(proba[pred])
    return {
        "label": "PortScan" if pred == 1 else "BENIGN",
        "confidence": round(confidence, 3),
    }

# ---------------------------------------------------------------
# PART 3 — A simple agent decision loop (rule-based stand-in for now)
# ---------------------------------------------------------------

def agent_decide(classification: dict, recent_hits_from_ip: int) -> dict:
    """
    Stand-in for the LLM reasoning step. Takes the classifier's output
    PLUS context (here: how many recent flows came from the same source
    IP) and decides an action. This is the exact interface an LLM agent
    would fill in later -- same inputs, same output shape, just smarter
    reasoning once wired to an actual model.
    """
    label = classification["label"]
    conf = classification["confidence"]

    if label == "BENIGN":
        return {"action": "log", "reason": "Flow classified as benign."}

    if conf > 0.9 and recent_hits_from_ip > 20:
        return {
            "action": "block_ip",
            "reason": (
                f"High-confidence PortScan ({conf:.0%}) with {recent_hits_from_ip} "
                "recent flows from the same source -- consistent with active recon."
            ),
        }
    elif conf > 0.7:
        return {
            "action": "alert_admin",
            "reason": f"PortScan detected with {conf:.0%} confidence -- needs review.",
        }
    else:
        return {
            "action": "log",
            "reason": f"Low-confidence PortScan signal ({conf:.0%}) -- logging for now.",
        }

# ---------------------------------------------------------------
# Demo run: simulate the agent reacting to 3 incoming flows
# ---------------------------------------------------------------

print("=== Agent demo on 3 sample flows from the test set ===\n")

sample_idx = X_test.sample(3, random_state=1).index
simulated_ip_hit_counts = [3, 27, 45]  # pretend context: recent hits per source IP

for i, (idx, hits) in enumerate(zip(sample_idx, simulated_ip_hit_counts), start=1):
    flow = X_test.loc[[idx]]
    result = classify_flow(flow)
    decision = agent_decide(result, hits)

    print(f"Flow {i}:")
    print(f"  classify_flow() -> {result}")
    print(f"  context: {hits} recent flows from this source IP")
    print(f"  agent decision -> action: {decision['action']}")
    print(f"                     reason: {decision['reason']}")
    print()