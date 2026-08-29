"""
Bayesian Network Based Heart Disease Risk Prediction
Structure Learning and Exact Inference from Clinical Data

Addis Ababa University, School of Information Technology and Engineering
Master's of Science in Artificial Intelligence
Probabilistic Graphical Models

By Fransi Ayele, ID: GSE/1254/18

Consolidated end-to-end pipeline: run this single script to reproduce every
result, figure, and metric reported in the accompanying report and notebook.

Usage:
    python heart_disease_bn_pipeline.py
"""

import sys
import os
import pickle

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import load_and_prepare
from structure_learning import learn_structure, expert_structure, fit_parameters
from inference import compare_inference_methods, evaluate_model
from sklearn.model_selection import train_test_split

import visualizations as viz

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "heart.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    print("=" * 70)
    print("STEP 1: Data loading and preprocessing")
    print("=" * 70)
    raw, data = load_and_prepare(DATA_PATH)
    print(f"Loaded {raw.shape[0]} patients, {raw.shape[1]} raw features.")
    print(f"Discretized to {data.shape[1]} categorical variables.")

    print("\n" + "=" * 70)
    print("STEP 2: Structure learning (full dataset, for the reported network)")
    print("=" * 70)
    print("Learning data-driven structure via Hill Climbing + BIC...")
    learned_model = learn_structure(data)
    learned_model = fit_parameters(learned_model, data)
    print("Learned edges:", list(learned_model.edges()))

    print("\nBuilding expert-specified structure...")
    expert_model = expert_structure()
    expert_model = fit_parameters(expert_model, data)
    print("Expert edges:", list(expert_model.edges()))

    print("\n" + "=" * 70)
    print("STEP 3: Exact inference comparison (Variable Elimination vs Junction Tree)")
    print("=" * 70)
    evidence = {"ChestPainType": "Asymptomatic", "ExerciseAngina": "Yes"}
    comparison = compare_inference_methods(expert_model, "HeartDisease", evidence, n_repeats=20)
    print(f"VE mean time: {comparison['ve_mean_time']*1000:.3f} ms")
    print(f"JT mean time: {comparison['bp_mean_time']*1000:.3f} ms")
    print(f"Max abs diff: {comparison['max_abs_diff']:.2e}")
    print(comparison["ve_result"])

    print("\n" + "=" * 70)
    print("STEP 4: Held-out evaluation (80/20 stratified split)")
    print("=" * 70)
    train_df, test_df = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["HeartDisease"]
    )
    learned_tt = fit_parameters(learn_structure(train_df), train_df)
    expert_tt = fit_parameters(expert_structure(), train_df)

    expert_eval = evaluate_model(expert_tt, test_df)
    learned_eval = evaluate_model(learned_tt, test_df)

    print("Expert-Structured Network:")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {k}: {expert_eval[k]:.4f}")
    print("Data-Learned Network:")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {k}: {learned_eval[k]:.4f}")

    results = {
        "expert_eval": expert_eval, "learned_eval": learned_eval,
        "comparison": comparison, "train_df": train_df, "test_df": test_df,
        "learned_model": learned_tt, "expert_model": expert_tt,
    }
    with open(os.path.join(OUT_DIR, "eval_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    print("\n" + "=" * 70)
    print("STEP 5: Generating figures")
    print("=" * 70)
    viz.fig1_target_distribution(raw)
    viz.fig2_age_distribution(raw)
    viz.fig3_missing_values(raw)
    viz.fig4_correlation_heatmap(raw)
    viz.fig5_expert_dag(expert_model)
    viz.fig6_learned_dag(learned_model)
    viz.fig7_inference_timing(comparison)
    viz.fig8_confusion_matrices(expert_eval, learned_eval)
    viz.fig9_model_comparison(expert_eval, learned_eval)
    viz.fig10_map_query_example(comparison["ve_result"])
    print("All figures written to", viz.FIGDIR)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
