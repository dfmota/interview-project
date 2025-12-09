from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


VALID_GROUND_TRUTHS = {"genuine", "fraud"}


def _compute_confusion_counts(df: pd.DataFrame) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix counts for the given DataFrame slice.

    We treat:
        - Positive class: ground_truth == "genuine"
        - Negative class: ground_truth == "fraud"
        - Predicted positive: decision == "accepted"
        - Predicted negative: decision == "rejected"

    Returns
    -------
    (tp, fn, fp, tn)
    """
    # Genuine (positive class)
    genuine = df["ground_truth"] == "genuine"
    # Fraud (negative class)
    fraud = df["ground_truth"] == "fraud"

    # Predictions
    accepted = df["decision"] == "accepted"
    rejected = df["decision"] == "rejected"

    tp = int((genuine & accepted).sum())
    fn = int((genuine & rejected).sum())
    fp = int((fraud & accepted).sum())
    tn = int((fraud & rejected).sum())

    return tp, fn, fp, tn


def _compute_metrics_from_counts(tp: int, fn: int, fp: int, tn: int) -> Dict[str, float]:
    """
    Given confusion matrix counts, compute accuracy, TPR and FPR.

    Metrics:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        TPR (recall for genuine) = TP / (TP + FN)
        FPR (fraud wrongly accepted) = FP / (FP + TN)
    """
    total = tp + fn + fp + tn

    accuracy = (tp + tn) / total if total > 0 else np.nan
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        "accuracy": float(accuracy),
        "tpr": float(tpr),
        "fpr": float(fpr),
    }


def compute_classification_metrics(
    df: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Compute overall and per-check_type classification metrics.

    The input DataFrame is expected to contain at least the following columns:
        - ground_truth: str, either "genuine" or "fraud"
        - decision: str, either "accepted" or "rejected"
        - check_type: str, either "id_document" or "selfie"

    Only rows where ground_truth is either "genuine" or "fraud" are considered.

    Metrics computed:
        - accuracy
        - TPR (recall for genuine) = TP / (TP + FN)
        - FPR (fraud wrongly accepted) = FP / (FP + TN)

    Returns
    -------
    overall_metrics : dict
        Dictionary with keys "accuracy", "tpr", "fpr".
    per_type_metrics : pd.DataFrame
        DataFrame with columns:
            - check_type
            - accuracy
            - tpr
            - fpr
    """
    # Filter out rows with invalid or missing ground truth labels
    valid_mask = df["ground_truth"].isin(VALID_GROUND_TRUTHS)
    df_valid = df.loc[valid_mask].copy()

    # Overall metrics
    tp, fn, fp, tn = _compute_confusion_counts(df_valid)
    overall_metrics = _compute_metrics_from_counts(tp, fn, fp, tn)

    # Per check_type metrics
    metrics_rows = []
    for check_type, df_group in df_valid.groupby("check_type"):
        tp_ct, fn_ct, fp_ct, tn_ct = _compute_confusion_counts(df_group)
        metrics = _compute_metrics_from_counts(tp_ct, fn_ct, fp_ct, tn_ct)
        metrics_rows.append(
            {
                "check_type": check_type,
                **metrics,
            }
        )

    per_type_metrics = pd.DataFrame(metrics_rows, columns=["check_type", "accuracy", "tpr", "fpr"])

    return overall_metrics, per_type_metrics
