"""
train_pipeline.py
-----------------
Modular training pipeline for the BalancedRandomForest classifier.

Pipeline stages
---------------
1. load_data          – read CSVs
2. load_baseline      – load saved baseline model & extract hyper-params
3. generate_oof_probs – 5-fold OOF probability generation
4. tune_thresholds    – per-class optimal thresholds via Youden's J
5. subset_features    – restrict to the deployment feature set
6. refit_model        – retrain on the full feature-subset training set
7. evaluate_model     – custom threshold-aware prediction + metrics
8. save_model         – persist the deployment package
"""

import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 345
N_SPLITS = 5
TARGET_COL = "CH"
BASELINE_MODEL_PATH = "models/baseline_model.pkl"
FINAL_MODEL_PATH = "models/final_model.pkl"
TRAIN_CSV = "path/to/training/data"
TEST_CSV = "path/to/test/data.csv"

COLS_TO_KEEP = [
    "RBC", "Hbconc", "MCV", "RDW", "platelet", "plateletcrit",
    "lymphocyte", "monocyte", "neutrophil", "eosinophil",
    "reticulocyte", "age", "giant_plt", "CH",
]

# ---------------------------------------------------------------------------
# Stage 1 – Data loading
# ---------------------------------------------------------------------------

def load_data(
    train_path: str = TRAIN_CSV,
    test_path: str = TEST_CSV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read training and test CSVs, dropping any stray index columns."""
    drop_unnamed = lambda col: col != "Unnamed: 0"  # noqa: E731
    training_df = pd.read_csv(train_path, usecols=drop_unnamed)
    test_df     = pd.read_csv(test_path,  usecols=drop_unnamed)
    print(f"Loaded training data : {training_df.shape}")
    print(f"Loaded test data     : {test_df.shape}")
    return training_df, test_df


# ---------------------------------------------------------------------------
# Stage 2 – Baseline model loading
# ---------------------------------------------------------------------------

def load_baseline(model_path: str = BASELINE_MODEL_PATH) -> tuple:
    """
    Load the pickled baseline package and return
    (loaded_model, best_params_RF).
    """
    print(f"\nLoading baseline model from '{model_path}' …")
    with open(model_path, "rb") as fh:
        package = pickle.load(fh)
    model = package["model"]
    params = model.get_params()
    print("Baseline hyper-parameters extracted.")
    return model, params


# ---------------------------------------------------------------------------
# Stage 3 – OOF probability generation
# ---------------------------------------------------------------------------

def generate_oof_probs(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a BalancedRandomForestClassifier with K-fold cross-validation and
    return out-of-fold probability matrix (n_samples × n_classes) and the
    sorted unique class array.
    """
    print(f"\nGenerating OOF probabilities ({n_splits}-fold) …")
    classes   = np.unique(y)
    n_classes = len(classes)
    oof_probs = np.zeros((X.shape[0], n_classes))

    kf    = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = BalancedRandomForestClassifier(**params)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr        = y.iloc[train_idx]

        model.fit(X_tr, y_tr)
        oof_probs[val_idx, :] = model.predict_proba(X_val)
        print(f"  Fold {fold}/{n_splits} complete.")

    return oof_probs, classes


# ---------------------------------------------------------------------------
# Stage 4 – Optimal threshold tuning (Youden's J)
# ---------------------------------------------------------------------------

def tune_thresholds(
    y: pd.Series,
    oof_probs: np.ndarray,
    classes: np.ndarray,
) -> dict:
    """
    Compute the one-vs-rest optimal threshold for each class using
    Youden's J statistic on the OOF probabilities.

    Returns an OrderedDict {class_label: threshold}.
    """
    print("\nDetermining optimal OvR thresholds (Youden's J) …")
    y_bin              = label_binarize(y, classes=classes)
    optimal_thresholds = OrderedDict()

    for i, cls in enumerate(classes):
        fpr, tpr, thresholds = roc_curve(y_bin[:, i], oof_probs[:, i])
        j_idx       = np.argmax(tpr - fpr)
        best_thresh = thresholds[j_idx]
        optimal_thresholds[cls] = best_thresh
        print(f"  Class {cls}: threshold = {best_thresh:.4f}")

    return optimal_thresholds


# ---------------------------------------------------------------------------
# Stage 5 – Feature subsetting
# ---------------------------------------------------------------------------

def subset_features(
    training_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols_to_keep: list[str] = COLS_TO_KEEP,
    target: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Restrict both DataFrames to `cols_to_keep` and split into
    (X_train, y_train, X_test, y_test).
    """
    print(f"\nSubsetting to {len(cols_to_keep) - 1} features …")
    train_sub = training_df[cols_to_keep]
    test_sub  = test_df[cols_to_keep]

    X_train = train_sub.drop(columns=[target])
    y_train = train_sub[target]
    X_test  = test_sub.drop(columns=[target])
    y_test  = test_sub[target]

    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Stage 6 – Final model refit
# ---------------------------------------------------------------------------

def refit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> BalancedRandomForestClassifier:
    """
    Retrain a fresh BalancedRandomForestClassifier on the full
    (feature-subsetted) training set.
    """
    print("\nRefitting final model on full training set …")
    model = BalancedRandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print("  Refit complete.")
    return model


# ---------------------------------------------------------------------------
# Stage 7 – Threshold-aware prediction & evaluation
# ---------------------------------------------------------------------------

def _predict_with_thresholds(
    y_proba: np.ndarray,
    classes: np.ndarray,
    threshold_array: np.ndarray,
) -> np.ndarray:
    """
    For each sample select the class with the largest (prob − threshold)
    margin among those that exceed their threshold; fall back to argmax
    probability when no class clears its threshold.
    """
    predictions = []
    for prob_vec in y_proba:
        passing = np.where(prob_vec >= threshold_array)[0]
        if passing.size > 0:
            margins     = prob_vec[passing] - threshold_array[passing]
            chosen_idx  = passing[np.argmax(margins)]
        else:
            chosen_idx  = np.argmax(prob_vec)
        predictions.append(classes[chosen_idx])
    return np.array(predictions)


def evaluate_model(
    model: BalancedRandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimal_thresholds: dict,
) -> np.ndarray:
    """
    Generate probability predictions, apply custom OvR thresholds,
    print evaluation metrics, and return the predicted label array.
    """
    print("\nEvaluating on test set …")
    classes         = np.array(sorted(optimal_thresholds.keys()))
    threshold_array = np.array([optimal_thresholds[c] for c in classes])

    y_proba    = model.predict_proba(X_test)
    y_pred     = _predict_with_thresholds(y_proba, classes, threshold_array)

    n_fallback = sum(
        np.all(pv < threshold_array) for pv in y_proba
    )
    print(f"  Fallback cases (no class cleared threshold): {n_fallback}")

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy (optimal OvR thresholds): {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred


# ---------------------------------------------------------------------------
# Stage 8 – Model persistence
# ---------------------------------------------------------------------------

def save_model(
    model: BalancedRandomForestClassifier,
    optimal_thresholds: dict,
    X_train: pd.DataFrame,
    save_path: str = FINAL_MODEL_PATH,
) -> None:
    """Persist the deployment package (model + thresholds + metadata)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    classes = np.unique(model.classes_)
    package = {
        "model":              model,
        "optimal_thresholds": optimal_thresholds,
        "classes":            classes.tolist(),
        "features":           X_train.columns.tolist(),
    }
    with open(save_path, "wb") as fh:
        pickle.dump(package, fh)
    print(f"\nDeployment package saved → '{save_path}'")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load raw data
    training_df, test_df = load_data()

    # 2. Load baseline model & hyper-parameters
    _, best_params = load_baseline()

    # 3. Generate OOF probabilities on the full feature set
    X_full_train = training_df.drop(columns=[TARGET_COL])
    y_full_train = training_df[TARGET_COL]
    oof_probs, classes = generate_oof_probs(X_full_train, y_full_train, best_params)

    # 4. Derive optimal per-class thresholds from OOF predictions
    optimal_thresholds = tune_thresholds(y_full_train, oof_probs, classes)

    # 5. Restrict to the deployment feature set
    X_train, y_train, X_test, y_test = subset_features(training_df, test_df)

    # 6. Refit the final model on the full (subsetted) training data
    final_model = refit_model(X_train, y_train, best_params)

    # 7. Evaluate on the held-out test set
    evaluate_model(final_model, X_test, y_test, optimal_thresholds)

    # 8. Persist the deployment package
    save_model(final_model, optimal_thresholds, X_train)


if __name__ == "__main__":
    main()