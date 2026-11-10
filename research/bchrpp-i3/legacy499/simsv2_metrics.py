"""Exploratory CH-SIMSv2 regression metrics; zero is positive for Acc-2."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def simsv2_metrics(prediction, label):
    pred = prediction.detach().cpu().numpy().reshape(-1).astype(np.float64)
    true = label.detach().cpu().numpy().reshape(-1).astype(np.float64)
    pred = np.clip(pred, -1.0, 1.0)
    true = np.clip(true, -1.0, 1.0)
    binary_pred, binary_true = pred >= 0, true >= 0
    three_pred = np.digitize(pred, [-0.1, 0.1], right=True)
    three_true = np.digitize(true, [-0.1, 0.1], right=True)
    five_pred = np.digitize(pred, [-0.7, -0.1, 0.1, 0.7], right=True)
    five_true = np.digitize(true, [-0.7, -0.1, 0.1, 0.7], right=True)
    corr = np.corrcoef(pred, true)[0, 1]
    return {
        "Mult_acc_2": round(float(accuracy_score(binary_true, binary_pred)), 4),
        "Mult_acc_3": round(float(accuracy_score(three_true, three_pred)), 4),
        "Mult_acc_5": round(float(accuracy_score(five_true, five_pred)), 4),
        "F1_score": round(float(f1_score(binary_true, binary_pred, average="weighted")), 4),
        "MAE": round(float(np.mean(np.abs(pred - true))), 4),
        "Corr": round(float(corr), 4),
    }
