# 🛡️ Agentic IDS Workflow

A multi-agent AI pipeline for network intrusion detection. Eight specialized agents autonomously load, normalize, train, evaluate, and self-correct — no manual steps between stages.

## Agents

| Agent | Role |
|---|---|
| `DatasetAgent` | Loads CSVs or generates synthetic network data |
| `SchemaAgent` | Normalizes heterogeneous datasets to a common 15-feature schema |
| `FeatureAgent` | Engineers derived features (throughput, login pressure, port risk, etc.) |
| `TrainerAgent` | Trains a Random Forest classifier (350 trees, balanced class weights) |
| `EvaluatorAgent` | Leave-one-dataset-out cross-evaluation |
| `TriageAgent` | Produces severity-tagged alerts with confidence + feature explanations |
| `FeedbackAgent` | Records analyst corrections on mispredictions |
| `RetrainAgent` | Retrains model with baseline data + feedback in a closed loop |

## Threat Classes

`benign` · `dos` · `probe` · `bruteforce` · `exfiltration` · `botnet`

Severity levels range from **informational** → **low** → **medium** → **high** → **critical**.

## Quickstart

```bash
pip install pandas numpy scikit-learn joblib
python agentic_ids_workflow.py
```

With your own dataset:

```bash
python agentic_ids_workflow.py --datasets '{"path":"data.csv","label_col":"Label","name":"my_network"}'
```

Outputs saved to `ids_artifacts/`:
- `ids_agentic_model.joblib` — trained model + schema metadata
- `evaluation_summary.json` — accuracy, F1 scores, case test results

## How It Works

```
DatasetAgent → SchemaAgent → FeatureAgent → TrainerAgent
                                                  ↓
                                          EvaluatorAgent (leave-one-out)
                                                  ↓
                                    TriageAgent → FeedbackAgent
                                                  ↓
                                          RetrainAgent → updated model
```

The feedback loop automatically collects mispredicted cases and retrains the model — no manual intervention needed.

## Requirements

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `joblib`
- No GPU required; runs on CPU (`n_jobs=-1` uses all cores)

## Dataset Format

Your CSV must have a label column (default: `label`) with values matching: `benign`, `dos`, `probe`, `bruteforce`, `exfiltration`, `botnet`. Unknown labels are remapped to `benign`. Missing feature columns are auto-filled.

## License

MIT
