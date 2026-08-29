# Heart Disease Risk Prediction (Bayesian Network)

This repository contains a Bayesian Network-based heart disease prediction system, developed as a course project for the Master's of Science in Artificial Intelligence at Addis Ababa University (Probabilistic Graphical Models).

## Overview

The project models the probabilistic relationships between clinical features and the presence of heart disease using the UCI Cleveland Heart Disease dataset. It implements:
1. **Data-driven Structure Learning**: Using Hill Climbing and BIC score to discover relationships from data.
2. **Expert-specified Structure**: A Bayesian Network grounded in cardiology domain knowledge.
3. **Exact Inference**: Using Variable Elimination and Junction Tree (Belief Propagation) algorithms via `pgmpy`.
4. **Interactive UI**: A Streamlit web application for interactive probabilistic queries.

## Project Structure

- `app.py`: Streamlit web application for interactive prediction.
- `heart_disease_bn_pipeline.py`: End-to-end pipeline script to reproduce all reported results and figures.
- `preprocessing.py`: Data loading and discretization logic.
- `structure_learning.py`: Functions for building the data-driven and expert Bayesian Networks, and fitting parameters (Dirichlet prior).
- `inference.py`: Wrappers for exact inference (Variable Elimination vs. Junction Tree) and model evaluation.
- `visualizations.py`: Code for generating plots and DAG visualizations.
- `data/heart.csv`: The UCI Cleveland Heart Disease dataset.

## Setup and Installation

### Prerequisites
- Python 3.11+
- Requirements: `pgmpy`, `streamlit`, `pandas`, `scikit-learn`, `numpy`

### Installation
1. Clone the repository or download the project files.
2. Install the required dependencies:
   ```bash
   pip install pgmpy streamlit pandas scikit-learn numpy
   ```
3. Ensure the dataset is located at `data/heart.csv`.

## Usage

### Running the Web Application
To launch the interactive predictor UI:
```bash
streamlit run app.py
```
This will open the application in your browser (default: `http://localhost:8501`), where you can input patient evidence and observe the computed probabilities for Heart Disease presence.

### Running the Full Pipeline
To run the complete analysis, evaluate the models, and generate all figures:
```bash
python heart_disease_bn_pipeline.py
```

## Disclaimer
This project is for educational and demonstrative purposes only (course project) and is **not intended as a medical diagnostic tool**.
