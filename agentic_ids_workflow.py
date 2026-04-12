
#!/usr/bin/env python3
"""
Intrusion Detection Agentic AI Workflow
======================================

A self-contained, runnable demo that simulates a multi-dataset intrusion
detection workflow with agent-style components:

- DatasetAgent: loads or generates multiple datasets
- SchemaAgent: aligns schemas across datasets
- FeatureAgent: engineers shared flow features
- TrainerAgent: trains a classifier
- EvaluatorAgent: evaluates cross-dataset performance
- TriageAgent: explains alerts and assigns severity
- FeedbackAgent: records analyst corrections
- RetrainAgent: retrains with feedback

The demo works out of the box with synthetic datasets that mimic different
network domains. It also includes a hook for plugging in CSV datasets.

Dependencies:
    pip install pandas numpy scikit-learn joblib
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RNG = np.random.default_rng(42)


LABELS = [
    "benign",
    "dos",
    "probe",
    "bruteforce",
    "exfiltration",
    "botnet",
]

SEVERITY_BY_LABEL = {
    "benign": "informational",
    "probe": "low",
    "botnet": "medium",
    "dos": "high",
    "bruteforce": "high",
    "exfiltration": "critical",
}


@dataclass
class DatasetArtifact:
    name: str
    frame: pd.DataFrame
    source_type: str = "synthetic"
    label_col: str = "label"


@dataclass
class WorkflowResult:
    holdout_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    report: str
    confusion: pd.DataFrame


class DatasetAgent:
    """Loads external CSV datasets or generates demo datasets."""

    def __init__(self, dataset_specs: Optional[List[dict]] = None):
        self.dataset_specs = dataset_specs or []

    def load(self) -> List[DatasetArtifact]:
        if self.dataset_specs:
            return [self._load_csv(spec) for spec in self.dataset_specs]
        return self._generate_demo_datasets()

    def _load_csv(self, spec: dict) -> DatasetArtifact:
        path = Path(spec["path"])
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)
        label_col = spec.get("label_col", "label")
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in {path}")

        name = spec.get("name", path.stem)
        source_type = spec.get("source_type", "csv")
        return DatasetArtifact(name=name, frame=df, source_type=source_type, label_col=label_col)

    def _generate_demo_datasets(self) -> List[DatasetArtifact]:
        specs = [
            ("cic_like", 3200, {"scale": 1.0, "shift": 0.0, "noise": 0.15}),
            ("unsw_like", 3200, {"scale": 1.35, "shift": 0.5, "noise": 0.25}),
            ("iot_like", 3200, {"scale": 0.75, "shift": -0.4, "noise": 0.12}),
        ]
        return [DatasetArtifact(name=name, frame=self._make_synthetic(name, n, cfg)) for name, n, cfg in specs]

    def _make_synthetic(self, name: str, n: int, cfg: dict) -> pd.DataFrame:
        classes = np.array(LABELS)
        weights = np.array([0.56, 0.12, 0.10, 0.09, 0.08, 0.05])
        y = RNG.choice(classes, size=n, p=weights)

        rows = []
        for label in y:
            rows.append(self._generate_row(label, cfg))
        df = pd.DataFrame(rows)
        df["label"] = y
        df["dataset_id"] = name

        # Dataset-specific skew to simulate real-world domain shift.
        if name == "unsw_like":
            df["bytes"] = np.clip(df["bytes"] * 1.25 + RNG.normal(20, 15, n), 1, None)
            df["packets"] = np.clip(df["packets"] * 1.10 + RNG.normal(0, 1.5, n), 1, None)
            df["syn_ratio"] = np.clip(df["syn_ratio"] + RNG.normal(0.02, 0.03, n), 0, 1)
        elif name == "iot_like":
            df["flow_duration"] = np.clip(df["flow_duration"] * 0.65 + RNG.normal(0, 0.25, n), 0.01, None)
            df["burstiness"] = np.clip(df["burstiness"] * 1.35 + RNG.normal(0, 0.05, n), 0, 1)
            df["dst_port"] = np.where(df["dst_port"] > 1024, RNG.choice([80, 443, 8080], size=n), df["dst_port"])

        # Missingness to ensure the pipeline is robust.
        for col in ["syn_ratio", "pkt_size_std", "failed_logins"]:
            idx = RNG.choice(n, size=max(1, n // 40), replace=False)
            df.loc[idx, col] = np.nan

        return df

    def _generate_row(self, label: str, cfg: dict) -> dict:
        scale = cfg["scale"]
        shift = cfg["shift"]
        noise = cfg["noise"]

        # Baseline benign traffic.
        flow_duration = abs(RNG.normal(1.6, 0.8)) * scale + 0.01
        packets = max(1.0, RNG.normal(18, 8) * scale + 2)
        bytes_ = max(40.0, RNG.normal(1700, 900) * scale + 100)
        syn_ratio = np.clip(RNG.normal(0.08, 0.05) + shift * 0.01, 0, 1)
        fin_ratio = np.clip(RNG.normal(0.20, 0.06), 0, 1)
        pkt_size_mean = max(64.0, RNG.normal(92, 24) * scale)
        pkt_size_std = max(5.0, RNG.normal(19, 7) * scale)
        entropy = np.clip(RNG.normal(0.35, 0.15), 0, 1)
        connection_count = max(1.0, RNG.normal(3, 2) * scale)
        failed_logins = max(0.0, RNG.poisson(0.15))
        burstiness = np.clip(RNG.normal(0.22, 0.10), 0, 1)
        src_rate = max(0.1, RNG.normal(8, 3) * scale)
        dst_rate = max(0.1, RNG.normal(7, 3) * scale)
        src_port = int(RNG.choice([20, 21, 22, 53, 80, 123, 443, 8080, 49152, 55000]))
        dst_port = int(RNG.choice([22, 53, 80, 110, 143, 443, 8080, 9001]))

        if label == "dos":
            packets *= RNG.uniform(5.0, 11.0)
            bytes_ *= RNG.uniform(4.0, 10.0)
            flow_duration *= RNG.uniform(0.3, 0.8)
            syn_ratio = np.clip(RNG.normal(0.80, 0.08), 0, 1)
            fin_ratio = np.clip(RNG.normal(0.03, 0.02), 0, 1)
            burstiness = np.clip(RNG.normal(0.85, 0.08), 0, 1)
            src_rate *= RNG.uniform(4.0, 8.0)
            dst_rate *= RNG.uniform(0.4, 1.0)
            dst_port = int(RNG.choice([80, 443, 8080, 53]))
        elif label == "probe":
            packets *= RNG.uniform(1.8, 4.5)
            bytes_ *= RNG.uniform(0.8, 1.8)
            flow_duration *= RNG.uniform(1.1, 2.6)
            entropy = np.clip(RNG.normal(0.82, 0.10), 0, 1)
            connection_count *= RNG.uniform(4.0, 10.0)
            src_rate *= RNG.uniform(1.8, 3.5)
            dst_port = int(RNG.choice([21, 22, 80, 139, 445, 8080, 3389]))
        elif label == "bruteforce":
            flow_duration *= RNG.uniform(1.8, 4.8)
            packets *= RNG.uniform(1.2, 2.2)
            bytes_ *= RNG.uniform(0.8, 1.4)
            failed_logins += RNG.poisson(8)
            connection_count *= RNG.uniform(2.0, 6.0)
            entropy = np.clip(RNG.normal(0.48, 0.11), 0, 1)
            src_rate *= RNG.uniform(1.1, 2.1)
            dst_port = int(RNG.choice([22, 23, 3389, 445]))
        elif label == "exfiltration":
            flow_duration *= RNG.uniform(2.5, 6.5)
            packets *= RNG.uniform(1.5, 3.5)
            bytes_ *= RNG.uniform(5.0, 13.0)
            pkt_size_mean *= RNG.uniform(1.3, 2.0)
            pkt_size_std *= RNG.uniform(1.1, 1.8)
            entropy = np.clip(RNG.normal(0.74, 0.12), 0, 1)
            dst_rate *= RNG.uniform(2.0, 5.0)
            src_rate *= RNG.uniform(1.3, 2.4)
            dst_port = int(RNG.choice([53, 80, 443, 8080, 8443]))
        elif label == "botnet":
            packets *= RNG.uniform(2.5, 6.0)
            bytes_ *= RNG.uniform(2.0, 5.0)
            flow_duration *= RNG.uniform(0.8, 2.0)
            syn_ratio = np.clip(RNG.normal(0.62, 0.10), 0, 1)
            fin_ratio = np.clip(RNG.normal(0.10, 0.04), 0, 1)
            burstiness = np.clip(RNG.normal(0.70, 0.11), 0, 1)
            connection_count *= RNG.uniform(5.0, 12.0)
            src_rate *= RNG.uniform(2.5, 5.0)
            dst_rate *= RNG.uniform(1.8, 4.0)
            dst_port = int(RNG.choice([25, 53, 80, 443, 6667, 8080]))

        pkt_size_mean = pkt_size_mean + np.log1p(bytes_ / max(packets, 1.0)) * 2.0
        pkt_size_std = max(1.0, pkt_size_std + RNG.normal(0, 2.0))
        entropy = np.clip(entropy + RNG.normal(0, noise), 0, 1)
        burstiness = np.clip(burstiness + RNG.normal(0, noise * 0.5), 0, 1)
        failed_logins = max(0.0, failed_logins + RNG.normal(0, noise))

        return {
            "flow_duration": float(max(0.01, flow_duration)),
            "packets": float(max(1.0, packets)),
            "bytes": float(max(1.0, bytes_)),
            "syn_ratio": float(np.clip(syn_ratio, 0, 1)),
            "fin_ratio": float(np.clip(fin_ratio, 0, 1)),
            "pkt_size_mean": float(pkt_size_mean),
            "pkt_size_std": float(pkt_size_std),
            "entropy": float(entropy),
            "connection_count": float(max(1.0, connection_count)),
            "failed_logins": float(max(0.0, failed_logins)),
            "burstiness": float(burstiness),
            "src_rate": float(max(0.1, src_rate)),
            "dst_rate": float(max(0.1, dst_rate)),
            "src_port": int(src_port),
            "dst_port": int(dst_port),
        }


class SchemaAgent:
    """Aligns heterogeneous datasets to a common schema."""

    NUMERIC_FEATURES = [
        "flow_duration",
        "packets",
        "bytes",
        "syn_ratio",
        "fin_ratio",
        "pkt_size_mean",
        "pkt_size_std",
        "entropy",
        "connection_count",
        "failed_logins",
        "burstiness",
        "src_rate",
        "dst_rate",
        "src_port",
        "dst_port",
    ]

    def normalize(self, artifact: DatasetArtifact) -> pd.DataFrame:
        df = artifact.frame.copy()
        if artifact.label_col != "label" and artifact.label_col in df.columns:
            df = df.rename(columns={artifact.label_col: "label"})
        if "dataset_id" not in df.columns:
            df["dataset_id"] = artifact.name
        for col in self.NUMERIC_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        if "label" not in df.columns:
            df["label"] = "benign"
        df["label"] = df["label"].astype(str).str.lower()
        df.loc[~df["label"].isin(LABELS), "label"] = "benign"
        df = df[["dataset_id"] + self.NUMERIC_FEATURES + ["label"]]
        return df


class FeatureAgent:
    """Creates derived features shared across all datasets."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["bytes_per_packet"] = out["bytes"] / out["packets"].replace(0, np.nan)
        out["packets_per_second"] = out["packets"] / out["flow_duration"].replace(0, np.nan)
        out["bytes_per_second"] = out["bytes"] / out["flow_duration"].replace(0, np.nan)
        out["log_bytes"] = np.log1p(out["bytes"])
        out["log_packets"] = np.log1p(out["packets"])
        out["login_pressure"] = out["failed_logins"] / (out["connection_count"] + 1.0)
        out["port_risk"] = np.where(
            out["dst_port"].isin([22, 23, 25, 53, 80, 110, 139, 143, 445, 3389, 8080, 8443]),
            1.0,
            0.0,
        )
        out["statefulness"] = (out["fin_ratio"] + out["syn_ratio"]) / 2.0
        out["traffic_skew"] = (out["src_rate"] - out["dst_rate"]) / (out["src_rate"] + out["dst_rate"] + 1.0)
        return out


class TrainerAgent:
    """Trains the core IDS model."""

    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.feature_columns: Optional[List[str]] = None

    def build_pipeline(self, feature_cols: List[str]) -> Pipeline:
        numeric_features = [c for c in feature_cols if c != "dataset_id"]
        categorical_features = ["dataset_id"]

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_features),
                ("cat", categorical_pipe, categorical_features),
            ],
            remainder="drop",
        )

        clf = RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

        return Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])

    def fit(self, df: pd.DataFrame) -> "TrainerAgent":
        feature_cols = [c for c in df.columns if c != "label"]
        self.feature_columns = feature_cols
        self.model = self.build_pipeline(feature_cols)
        self.model.fit(df[feature_cols], df["label"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_trained()
        return self.model.predict(df[self.feature_columns])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._ensure_trained()
        return self.model.predict_proba(df[self.feature_columns])

    def feature_importances(self) -> Dict[str, float]:
        self._ensure_trained()
        pre = self.model.named_steps["preprocessor"]
        clf = self.model.named_steps["clf"]

        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][2]

        feature_names = list(num_cols)
        if len(cat_cols):
            ohe = pre.named_transformers_["cat"].named_steps["onehot"]
            cat_names = list(ohe.get_feature_names_out(cat_cols))
            feature_names.extend(cat_names)

        importances = clf.feature_importances_
        ordered = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return dict(ordered)

    def _ensure_trained(self):
        if self.model is None or self.feature_columns is None:
            raise RuntimeError("Model is not trained yet.")


class EvaluatorAgent:
    """Scores models on cross-dataset holdouts."""

    def evaluate(self, trainer: TrainerAgent, test_df: pd.DataFrame, holdout_name: str) -> WorkflowResult:
        y_true = test_df["label"].to_numpy()
        y_pred = trainer.predict(test_df)

        acc = accuracy_score(y_true, y_pred)
        macro = f1_score(y_true, y_pred, average="macro")
        weighted = f1_score(y_true, y_pred, average="weighted")
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)

        labels_order = LABELS
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels_order], columns=[f"pred_{x}" for x in labels_order])

        return WorkflowResult(
            holdout_name=holdout_name,
            accuracy=acc,
            macro_f1=macro,
            weighted_f1=weighted,
            report=report,
            confusion=cm_df,
        )


class TriageAgent:
    """Turns predictions into analyst-friendly incident summaries."""

    def __init__(self, trainer: TrainerAgent, schema_agent: SchemaAgent, feature_agent: FeatureAgent):
        self.trainer = trainer
        self.schema_agent = schema_agent
        self.feature_agent = feature_agent

    def triage(self, sample: pd.DataFrame) -> dict:
        prepared = self._prepare(sample)
        pred = self.trainer.predict(prepared)[0]
        proba = self.trainer.predict_proba(prepared)[0]
        labels = list(self.trainer.model.named_steps["clf"].classes_)
        score_map = {label: float(prob) for label, prob in zip(labels, proba)}

        features = prepared.iloc[0].to_dict()
        top_feats = self._top_features(features)

        return {
            "predicted_label": pred,
            "severity": SEVERITY_BY_LABEL.get(pred, "unknown"),
            "confidence": score_map.get(pred, max(score_map.values())),
            "top_contributing_features": top_feats,
            "raw_probabilities": score_map,
        }

    def _prepare(self, sample: pd.DataFrame) -> pd.DataFrame:
        sample = sample.copy()
        if "dataset_id" not in sample.columns:
            sample["dataset_id"] = "demo"
        artifact = DatasetArtifact(name=str(sample.iloc[0].get("dataset_id", "demo")), frame=sample)
        normalized = self.schema_agent.normalize(artifact)
        return self.feature_agent.transform(normalized)

    def _top_features(self, features: dict, top_k: int = 5) -> List[Tuple[str, float]]:
        importances = self.trainer.feature_importances()
        normalized = []
        for name, importance in list(importances.items())[:top_k * 2]:
            value = features.get(name, np.nan)
            if pd.isna(value):
                score = importance
            else:
                score = float(abs(value)) * float(importance)
            normalized.append((name, score))
        normalized.sort(key=lambda x: x[1], reverse=True)
        return normalized[:top_k]


class FeedbackAgent:
    """Stores analyst corrections for later retraining."""

    def __init__(self):
        self.samples: List[pd.DataFrame] = []
        self.labels: List[str] = []

    def add(self, sample: pd.DataFrame, corrected_label: str):
        self.samples.append(sample.copy())
        self.labels.append(corrected_label)

    def has_data(self) -> bool:
        return len(self.samples) > 0

    def build_frame(self) -> pd.DataFrame:
        if not self.has_data():
            raise RuntimeError("No feedback samples available.")
        df = pd.concat(self.samples, ignore_index=True)
        df["label"] = self.labels
        return df


class RetrainAgent:
    """Retrains the model with baseline data plus feedback cases."""

    def retrain(self, trainer: TrainerAgent, base_df: pd.DataFrame, feedback_df: pd.DataFrame) -> TrainerAgent:
        combined = pd.concat([base_df, feedback_df], ignore_index=True)
        trainer.fit(combined)
        return trainer


def prepare_working_frame(raw_df: pd.DataFrame, schema_agent: SchemaAgent, feature_agent: FeatureAgent, artifact_name: str) -> pd.DataFrame:
    artifact = DatasetArtifact(name=artifact_name, frame=raw_df)
    normalized = schema_agent.normalize(artifact)
    enriched = feature_agent.transform(normalized)
    return enriched


def make_case(**kwargs) -> pd.DataFrame:
    base = {
        "dataset_id": kwargs.pop("dataset_id", "demo"),
        "flow_duration": kwargs.pop("flow_duration", 1.2),
        "packets": kwargs.pop("packets", 20.0),
        "bytes": kwargs.pop("bytes", 1600.0),
        "syn_ratio": kwargs.pop("syn_ratio", 0.05),
        "fin_ratio": kwargs.pop("fin_ratio", 0.2),
        "pkt_size_mean": kwargs.pop("pkt_size_mean", 90.0),
        "pkt_size_std": kwargs.pop("pkt_size_std", 18.0),
        "entropy": kwargs.pop("entropy", 0.35),
        "connection_count": kwargs.pop("connection_count", 2.0),
        "failed_logins": kwargs.pop("failed_logins", 0.0),
        "burstiness": kwargs.pop("burstiness", 0.2),
        "src_rate": kwargs.pop("src_rate", 8.0),
        "dst_rate": kwargs.pop("dst_rate", 7.0),
        "src_port": kwargs.pop("src_port", 50000),
        "dst_port": kwargs.pop("dst_port", 443),
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def run_case_tests(triage_agent: TriageAgent) -> List[dict]:
    cases = [
        ("benign", make_case()),
        ("dos", make_case(flow_duration=0.25, packets=180, bytes=18000, syn_ratio=0.91, fin_ratio=0.01, burstiness=0.94, src_rate=60, dst_rate=4, dst_port=80)),
        ("probe", make_case(flow_duration=3.0, packets=70, bytes=2400, entropy=0.89, connection_count=40, src_rate=20, dst_port=445)),
        ("bruteforce", make_case(flow_duration=4.2, packets=34, bytes=1900, failed_logins=14, connection_count=22, entropy=0.5, dst_port=22)),
        ("exfiltration", make_case(flow_duration=5.0, packets=55, bytes=25000, pkt_size_mean=180, pkt_size_std=40, entropy=0.78, dst_rate=20, dst_port=443)),
        ("botnet", make_case(flow_duration=1.2, packets=90, bytes=11000, syn_ratio=0.7, fin_ratio=0.08, burstiness=0.8, connection_count=35, src_rate=28, dst_rate=16, dst_port=8080)),
    ]

    results = []
    for expected, sample in cases:
        triaged = triage_agent.triage(sample)
        results.append({"expected": expected, "predicted": triaged["predicted_label"], "severity": triaged["severity"], "confidence": triaged["confidence"]})
    return results


def print_case_summary(results: List[dict]):
    print("\nCase tests")
    print("-" * 70)
    for idx, row in enumerate(results, 1):
        status = "OK" if row["expected"] == row["predicted"] else "MISMATCH"
        print(f"{idx:>2}. expected={row['expected']:<13} predicted={row['predicted']:<13} severity={row['severity']:<12} confidence={row['confidence']:.3f} {status}")


def main():
    parser = argparse.ArgumentParser(description="Agentic AI IDS workflow demo")
    parser.add_argument("--save-dir", default="ids_artifacts", help="Directory to save the trained model and reports")
    parser.add_argument("--datasets", nargs="*", help="Optional JSON strings or CSV paths for external datasets")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_specs = []
    if args.datasets:
        for item in args.datasets:
            if item.strip().startswith("{"):
                dataset_specs.append(json.loads(item))
            else:
                dataset_specs.append({"path": item})

    dataset_agent = DatasetAgent(dataset_specs=dataset_specs or None)
    schema_agent = SchemaAgent()
    feature_agent = FeatureAgent()
    trainer = TrainerAgent()
    evaluator = EvaluatorAgent()
    feedback_agent = FeedbackAgent()
    retrain_agent = RetrainAgent()

    datasets = dataset_agent.load()
    prepared: Dict[str, pd.DataFrame] = {}

    for artifact in datasets:
        prepared[artifact.name] = prepare_working_frame(artifact.frame, schema_agent, feature_agent, artifact.name)

    # Cross-dataset evaluation: hold out each dataset once.
    results: List[WorkflowResult] = []
    for holdout_name in prepared.keys():
        train_frames = [df for name, df in prepared.items() if name != holdout_name]
        train_df = pd.concat(train_frames, ignore_index=True)
        test_df = prepared[holdout_name].copy()

        # Make sure features are consistent before training.
        trainer.fit(train_df)
        result = evaluator.evaluate(trainer, test_df, holdout_name)
        results.append(result)

    # Train final model on everything.
    all_data = pd.concat(list(prepared.values()), ignore_index=True)
    trainer.fit(all_data)

    triage_agent = TriageAgent(trainer, schema_agent, feature_agent)
    case_results = run_case_tests(triage_agent)

    # Collect wrong cases as analyst feedback and retrain once.
    for expected, result in zip(
        ["benign", "dos", "probe", "bruteforce", "exfiltration", "botnet"],
        case_results,
    ):
        if result["expected"] != result["predicted"]:
            # Create a deterministic correction sample.
            sample = make_case()
            if expected == "dos":
                sample = make_case(flow_duration=0.25, packets=180, bytes=18000, syn_ratio=0.91, fin_ratio=0.01, burstiness=0.94, src_rate=60, dst_rate=4, dst_port=80)
            elif expected == "probe":
                sample = make_case(flow_duration=3.0, packets=70, bytes=2400, entropy=0.89, connection_count=40, src_rate=20, dst_port=445)
            elif expected == "bruteforce":
                sample = make_case(flow_duration=4.2, packets=34, bytes=1900, failed_logins=14, connection_count=22, entropy=0.5, dst_port=22)
            elif expected == "exfiltration":
                sample = make_case(flow_duration=5.0, packets=55, bytes=25000, pkt_size_mean=180, pkt_size_std=40, entropy=0.78, dst_rate=20, dst_port=443)
            elif expected == "botnet":
                sample = make_case(flow_duration=1.2, packets=90, bytes=11000, syn_ratio=0.7, fin_ratio=0.08, burstiness=0.8, connection_count=35, src_rate=28, dst_rate=16, dst_port=8080)
            feedback_agent.add(sample, expected)

    retrained = False
    if feedback_agent.has_data():
        feedback_df = feature_agent.transform(schema_agent.normalize(DatasetArtifact(name="feedback", frame=feedback_agent.build_frame())))
        trainer = retrain_agent.retrain(trainer, all_data, feedback_df)
        triage_agent = TriageAgent(trainer, schema_agent, feature_agent)
        case_results = run_case_tests(triage_agent)
        retrained = True

    # Persist artifacts.
    model_path = save_dir / "ids_agentic_model.joblib"
    joblib.dump(
        {
            "trainer": trainer,
            "schema_features": schema_agent.NUMERIC_FEATURES,
            "labels": LABELS,
            "retrained": retrained,
        },
        model_path,
    )

    summary_path = save_dir / "evaluation_summary.json"
    summary = {
        "cross_dataset": [
            {
                "holdout_name": r.holdout_name,
                "accuracy": round(r.accuracy, 4),
                "macro_f1": round(r.macro_f1, 4),
                "weighted_f1": round(r.weighted_f1, 4),
            }
            for r in results
        ],
        "case_tests": case_results,
        "retrained": retrained,
        "model_path": str(model_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nAgentic IDS workflow completed.")
    print("=" * 72)

    for r in results:
        print(f"\nHoldout dataset: {r.holdout_name}")
        print(f"Accuracy:   {r.accuracy:.4f}")
        print(f"Macro F1:   {r.macro_f1:.4f}")
        print(f"Weighted F1:{r.weighted_f1:.4f}")
        print(r.report)
        print("Confusion matrix:")
        print(r.confusion)

    print_case_summary(case_results)

    print("\nSaved model to:", model_path)
    print("Saved summary to:", summary_path)

    if retrained:
        print("\nFeedback loop triggered and retraining completed.")


if __name__ == "__main__":
    main()
