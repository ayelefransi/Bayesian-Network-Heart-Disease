"""
Generates all figures for the Heart Disease Bayesian Network report.
Palette restricted to black / navy / steel-blue per style guide.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import sys
sys.path.insert(0, ".")
from preprocessing import load_and_prepare

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

NAVY = "#0B1F3A"
STEEL = "#4A6FA5"
BLACK = "#111111"
LIGHT_STEEL = "#A9C0DE"
GREY = "#6B6B6B"

FIGDIR = "figures"


def fig1_target_distribution(raw):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    counts = raw["HeartDisease"].map({1: "Present", 0: "Absent"}).value_counts()
    ax.bar(counts.index, counts.values, color=[NAVY, STEEL], edgecolor=BLACK, width=0.55)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center", fontsize=11, color=BLACK)
    ax.set_title("Figure 1. Distribution of Heart Disease Diagnosis", fontsize=12)
    ax.set_ylabel("Number of Patients")
    ax.set_ylim(0, max(counts.values) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig1_target_distribution.png", dpi=200)
    plt.close()


def fig2_age_distribution(raw):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    disease = raw[raw["HeartDisease"] == 1]["Age"]
    healthy = raw[raw["HeartDisease"] == 0]["Age"]
    bins = np.arange(25, 80, 5)
    ax.hist(healthy, bins=bins, alpha=0.75, label="Absent", color=LIGHT_STEEL, edgecolor=BLACK)
    ax.hist(disease, bins=bins, alpha=0.85, label="Present", color=NAVY, edgecolor=BLACK)
    ax.set_title("Figure 2. Age Distribution by Diagnosis", fontsize=12)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Patients")
    ax.legend(title="Heart Disease")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig2_age_distribution.png", dpi=200)
    plt.close()


def fig3_missing_values(raw):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    missing = raw.isna().sum()
    ax.bar(missing.index, missing.values, color=STEEL, edgecolor=BLACK)
    ax.set_title("Figure 3. Missing Value Count per Feature", fontsize=12)
    ax.set_ylabel("Missing Count")
    ax.set_xticklabels(missing.index, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "No missing values in this dataset (0 / 303 records)",
            transform=ax.transAxes, ha="center", va="center", fontsize=10,
            color=BLACK, style="italic")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig3_missing_values.png", dpi=200)
    plt.close()


def fig4_correlation_heatmap(raw):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    corr = raw.corr(numeric_only=True)
    im = ax.imshow(corr.values, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color=BLACK)
    ax.set_title("Figure 4. Feature Correlation Matrix", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig4_correlation_heatmap.png", dpi=200)
    plt.close()


def _draw_dag(model, title, filename, node_colors=None):
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    if "HeartDisease" in G.nodes():
        pos = nx.spring_layout(G, seed=11, k=2.2, iterations=200)
    else:
        pos = nx.spring_layout(G, seed=11, k=1.8, iterations=200)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    if node_colors is None:
        colors = [NAVY if n == "HeartDisease" else STEEL for n in G.nodes()]
    else:
        colors = node_colors
    node_size = 900

    nx.draw_networkx_edges(G, pos, edge_color=GREY, arrows=True, arrowsize=16,
                            arrowstyle="-|>", connectionstyle="arc3,rad=0.08", ax=ax,
                            node_size=node_size, width=1.3)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_size, ax=ax,
                            edgecolors=BLACK, linewidths=1.2)

    # Place labels just above each node, outside the circle, with a light backdrop
    label_pos = {k: (v[0], v[1] + 0.08) for k, v in pos.items()}
    for node, (x, y) in label_pos.items():
        ax.text(x, y, node, fontsize=9, ha="center", va="bottom", color=BLACK,
                family="serif", fontweight="bold" if node == "HeartDisease" else "normal",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0))

    ax.set_title(title, fontsize=13)
    ax.axis("off")
    ax.margins(0.15)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/{filename}", dpi=200)
    plt.close()


def fig5_expert_dag(expert_model):
    _draw_dag(expert_model, "Figure 5. Expert-Specified Bayesian Network Structure",
              "fig5_expert_dag.png")


def fig6_learned_dag(learned_model):
    _draw_dag(learned_model, "Figure 6. Data-Learned Bayesian Network Structure (Hill Climbing, BIC)",
              "fig6_learned_dag.png")


def fig7_inference_timing(comparison):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    methods = ["Variable\nElimination", "Junction Tree\n(Belief Propagation)"]
    means = [comparison["ve_mean_time"] * 1000, comparison["bp_mean_time"] * 1000]
    stds = [comparison["ve_std_time"] * 1000, comparison["bp_std_time"] * 1000]
    ax.bar(methods, means, yerr=stds, color=[NAVY, STEEL], edgecolor=BLACK, capsize=6, width=0.5)
    ax.set_title("Figure 7. Exact Inference Runtime Comparison", fontsize=12)
    ax.set_ylabel("Mean Query Time (ms, 20 runs)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig7_inference_timing.png", dpi=200)
    plt.close()


def fig8_confusion_matrices(expert_eval, learned_eval):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (name, ev) in zip(axes, [("Expert-Structured", expert_eval), ("Data-Learned", learned_eval)]):
        cm = ev["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(ev["labels"]); ax.set_yticklabels(ev["labels"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else BLACK)
        ax.set_title(name, fontsize=11)
    fig.suptitle("Figure 8. Confusion Matrices on Held-out Test Set (MAP Inference)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig8_confusion_matrices.png", dpi=200)
    plt.close()


def fig9_model_comparison(expert_eval, learned_eval):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.35
    expert_vals = [expert_eval[m] for m in metrics]
    learned_vals = [learned_eval[m] for m in metrics]
    ax.bar(x - width/2, expert_vals, width, label="Expert-Structured", color=NAVY, edgecolor=BLACK)
    ax.bar(x + width/2, learned_vals, width, label="Data-Learned", color=STEEL, edgecolor=BLACK)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 9. Model Performance Comparison", fontsize=12)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig9_model_comparison.png", dpi=200)
    plt.close()


def fig10_map_query_example(ve_result):
    fig, ax = plt.subplots(figsize=(6, 4.0))
    states = ve_result.state_names[ve_result.variables[0]]
    values = ve_result.values
    ax.bar(states, values, color=[NAVY, STEEL], edgecolor=BLACK, width=0.5)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)
    ax.set_title("Figure 10. Example Query: P(HeartDisease | ChestPain=Asymptomatic,\nExerciseAngina=Yes)", fontsize=11)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig10_map_query_example.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    raw, data = load_and_prepare("data/heart.csv")
    with open("data/eval_results.pkl", "rb") as f:
        results = pickle.load(f)

    fig1_target_distribution(raw)
    fig2_age_distribution(raw)
    fig3_missing_values(raw)
    fig4_correlation_heatmap(raw)
    fig5_expert_dag(results["expert_model"])
    fig6_learned_dag(results["learned_model"])
    fig7_inference_timing(results["comparison"])
    fig8_confusion_matrices(results["expert_eval"], results["learned_eval"])
    fig9_model_comparison(results["expert_eval"], results["learned_eval"])
    fig10_map_query_example(results["comparison"]["ve_result"])

    print("All figures generated in", FIGDIR)
