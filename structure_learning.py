"""
Bayesian Network structure and parameter learning for the Heart Disease project.

Builds two networks over the same discretized data:
  1. A data-driven structure learned with Hill Climbing + BIC score.
  2. An expert-specified structure grounded in cardiology domain knowledge.

Both are parameterized with Bayesian (Dirichlet-prior) estimation and saved
for downstream inference.
"""

import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BIC, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
import pickle
import sys
sys.path.insert(0, ".")
from preprocessing import load_and_prepare


def learn_structure(data, random_seed=42):
    """Learn DAG structure from data using Hill Climbing with BIC score."""
    np.random.seed(random_seed)
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=BIC(data), max_indegree=4, max_iter=int(1e4))
    model = DiscreteBayesianNetwork(best_model.edges())
    model.add_nodes_from(data.columns)
    return model


def expert_structure():
    """
    Expert-specified Bayesian Network grounded in cardiology domain knowledge.

    Rationale:
      - Age influences resting BP, cholesterol, max heart rate, and vessel
        calcification (MajorVessels), consistent with cardiovascular aging.
      - Sex influences cholesterol profile and chest pain presentation.
      - RestingBP and Cholesterol are classic modifiable risk factors feeding
        directly into HeartDisease.
      - HeartDisease is the common cause of the exercise-test findings:
        ExerciseAngina, STDepression, STSlope, MaxHeartRate response,
        ChestPainType, MajorVessels, and Thalassemia results (this mirrors
        how these are diagnostic *consequences/markers* observed during
        work-up, not independent causes).
      - FastingBloodSugar and RestingECG are conditionally linked to Age.
    """
    edges = [
        ("Age", "RestingBP"),
        ("Age", "Cholesterol"),
        ("Age", "MaxHeartRate"),
        ("Age", "FastingBloodSugar"),
        ("Sex", "Cholesterol"),
        ("Sex", "ChestPainType"),
        ("RestingBP", "HeartDisease"),
        ("Cholesterol", "HeartDisease"),
        ("FastingBloodSugar", "HeartDisease"),
        ("Age", "HeartDisease"),
        ("HeartDisease", "ChestPainType"),
        ("HeartDisease", "ExerciseAngina"),
        ("HeartDisease", "STDepression"),
        ("HeartDisease", "STSlope"),
        ("HeartDisease", "MaxHeartRate"),
        ("HeartDisease", "MajorVessels"),
        ("HeartDisease", "Thalassemia"),
        ("HeartDisease", "RestingECG"),
    ]
    model = DiscreteBayesianNetwork(edges)
    return model


def fit_parameters(model, data, prior="BDeu", equivalent_sample_size=10):
    """Estimate CPDs with Bayesian estimation (Dirichlet prior) to avoid zero counts."""
    estimator = BayesianEstimator(model, data)
    cpds = estimator.get_parameters(
        prior_type=prior, equivalent_sample_size=equivalent_sample_size
    )
    model.add_cpds(*cpds)
    assert model.check_model()
    return model


if __name__ == "__main__":
    raw, data = load_and_prepare("data/heart.csv")

    print("Learning data-driven structure (Hill Climbing + BIC)...")
    learned = learn_structure(data)
    print("Learned edges:", list(learned.edges()))
    learned = fit_parameters(learned, data)

    print("\nBuilding expert-specified structure...")
    expert = expert_structure()
    print("Expert edges:", list(expert.edges()))
    expert = fit_parameters(expert, data)

    with open("data/learned_model.pkl", "wb") as f:
        pickle.dump(learned, f)
    with open("data/expert_model.pkl", "wb") as f:
        pickle.dump(expert, f)
    data.to_pickle("data/clean_data.pkl")
    raw.to_pickle("data/raw_data.pkl")

    print("\nModels fit and saved.")
    print("Learned model nodes:", learned.number_of_nodes(), "edges:", learned.number_of_edges())
    print("Expert model nodes:", expert.number_of_nodes(), "edges:", expert.number_of_edges())
