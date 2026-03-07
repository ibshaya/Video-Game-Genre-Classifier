"""
Ensemble methods: weighted averaging and stacking.

Stacking trains a per-class Logistic Regression meta-learner on
out-of-fold probabilities from all base models.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def weighted_average(probs_list, weights):
    """Blend multiple probability arrays with given weights."""
    assert len(probs_list) == len(weights)
    result = np.zeros_like(probs_list[0])
    for probs, w in zip(probs_list, weights):
        result += w * probs
    return result


def stack_predictions(oof_probs_list, Y, test_probs_list, labels, threshold=0.5):
    """
    Per-class Logistic Regression stacking.

    Args:
        oof_probs_list: List of OOF probability arrays [shape (N, C) each].
        Y: Ground truth labels, shape (N, C).
        test_probs_list: List of test probability arrays [shape (M, C) each].
        labels: List of label names.
        threshold: Classification threshold.

    Returns:
        test_preds: Hard predictions for test set, shape (M, C).
        test_probs: Stacked probabilities for test set, shape (M, C).
        oof_f1: OOF macro F1 score.
    """
    C = len(labels)
    N = Y.shape[0]
    M = test_probs_list[0].shape[0]

    oof_meta = np.zeros((N, C), dtype=float)
    test_meta = np.zeros((M, C), dtype=float)

    for c in range(C):
        # Stack OOF probabilities from all models as features
        X_train = np.column_stack([oof[:, c] for oof in oof_probs_list])
        y_train = Y[:, c].astype(int)

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        oof_meta[:, c] = clf.predict_proba(X_train)[:, 1]

        X_test = np.column_stack([tp[:, c] for tp in test_probs_list])
        test_meta[:, c] = clf.predict_proba(X_test)[:, 1]

    # Evaluate OOF
    oof_preds = (oof_meta >= threshold).astype(int)
    oof_f1 = f1_score(Y, oof_preds, average="macro", zero_division=0)

    # Final predictions
    test_preds = (test_meta >= threshold).astype(int)

    return test_preds, test_meta, oof_f1
