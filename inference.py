"""
Exact inference and evaluation for the Heart Disease Bayesian Network project.

Provides:
  - Variable Elimination and Junction Tree query wrappers with timing.
  - A held-out evaluation routine that predicts HeartDisease presence/absence
    via MAP inference and reports accuracy, precision, recall, F1, and a
    confusion matrix.
"""

import time
import pickle
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination, BeliefPropagation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import sys
sys.path.insert(0, ".")
from preprocessing import load_and_prepare
from structure_learning import fit_parameters, expert_structure, learn_structure


def timed_query(inference_engine, variables, evidence):
    start = time.perf_counter()
    result = inference_engine.query(variables=variables, evidence=evidence, show_progress=False)
    elapsed = time.perf_counter() - start
    return result, elapsed


def compare_inference_methods(model, query_var, evidence, n_repeats=20):
    """Run the same query with Variable Elimination and Junction Tree (Belief Propagation),
    repeating to get a stable average runtime, and confirm both produce matching results."""
    ve = VariableElimination(model)
    bp = BeliefPropagation(model)  # uses a Junction Tree internally

    ve_times, bp_times = [], []
    for _ in range(n_repeats):
        _, t = timed_query(ve, [query_var], evidence)
        ve_times.append(t)
        _, t = timed_query(bp, [query_var], evidence)
        bp_times.append(t)

    ve_result, _ = timed_query(ve, [query_var], evidence)
    bp_result, _ = timed_query(bp, [query_var], evidence)

    return {
        "ve_result": ve_result,
        "bp_result": bp_result,
        "ve_mean_time": float(np.mean(ve_times)),
        "bp_mean_time": float(np.mean(bp_times)),
        "ve_std_time": float(np.std(ve_times)),
        "bp_std_time": float(np.std(bp_times)),
        "max_abs_diff": float(np.max(np.abs(ve_result.values - bp_result.values))),
    }


def evaluate_model(model, test_df, target="HeartDisease", method="VE"):
    """
    Predict HeartDisease for every row in test_df using MAP inference over
    all other observed variables, then compute classification metrics.
    """
    infer = VariableElimination(model) if method == "VE" else BeliefPropagation(model)

    y_true, y_pred = [], []
    model_nodes = set(model.nodes())

    for _, row in test_df.iterrows():
        evidence = {
            col: row[col] for col in test_df.columns
            if col != target and col in model_nodes
        }
        try:
            map_result = infer.map_query(variables=[target], evidence=evidence, show_progress=False)
            pred = map_result[target]
        except Exception:
            pred = "Absent"  # fallback if evidence state unseen in training fold
        y_true.append(row[target])
        y_pred.append(pred)

    labels = ["Present", "Absent"]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="Present", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="Present", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="Present", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm, "labels": labels,
        "y_true": y_true, "y_pred": y_pred,
    }


if __name__ == "__main__":
    raw, data = load_and_prepare("data/heart.csv")
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data["HeartDisease"])

    print("Refitting models on the training split only (for honest held-out evaluation)...")
    learned = learn_structure(train_df)
    learned = fit_parameters(learned, train_df)
    expert = expert_structure()
    expert = fit_parameters(expert, train_df)

    print("\n=== Inference Method Comparison (Variable Elimination vs Junction Tree) ===")
    comparison = compare_inference_methods(
        expert, "HeartDisease",
        {"ChestPainType": "Asymptomatic", "ExerciseAngina": "Yes"}
    )
    print(f"VE mean time:  {comparison['ve_mean_time']*1000:.3f} ms (+/- {comparison['ve_std_time']*1000:.3f})")
    print(f"JT mean time:  {comparison['bp_mean_time']*1000:.3f} ms (+/- {comparison['bp_std_time']*1000:.3f})")
    print(f"Max abs diff between methods: {comparison['max_abs_diff']:.2e}")
    print(comparison["ve_result"])

    print("\n=== Held-out Evaluation: Expert-Structured Network ===")
    expert_eval = evaluate_model(expert, test_df)
    print({k: v for k, v in expert_eval.items() if k not in ("y_true", "y_pred")})

    print("\n=== Held-out Evaluation: Learned Network ===")
    learned_eval = evaluate_model(learned, test_df)
    print({k: v for k, v in learned_eval.items() if k not in ("y_true", "y_pred")})

    with open("data/eval_results.pkl", "wb") as f:
        pickle.dump({
            "expert_eval": expert_eval, "learned_eval": learned_eval,
            "comparison": comparison, "train_df": train_df, "test_df": test_df,
            "learned_model": learned, "expert_model": expert,
        }, f)
    print("\nSaved evaluation results.")
