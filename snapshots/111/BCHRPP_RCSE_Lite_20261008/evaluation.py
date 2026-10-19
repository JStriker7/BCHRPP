"""Sentiment regression benchmark metrics; official split conventions vary."""
import numpy as np


def _corr(p, y):
    return float(np.corrcoef(p, y)[0, 1]) if len(y)>1 and np.std(p)>1e-12 and np.std(y)>1e-12 else 0.0


def _weighted_f1(y, p):
    # Binary weighted F1; no sklearn runtime dependency.
    total = len(y)
    result = 0.
    for cls in np.unique(y):
        tp = np.sum((y == cls) & (p == cls))
        precision_den = np.sum(p == cls)
        recall_den = np.sum(y == cls)
        precision = tp / max(precision_den, 1)
        recall = tp / max(recall_den, 1)
        f1 = 2*precision*recall / max(precision+recall, 1e-12)
        result += f1 * recall_den / total
    return float(result)


def mosi_metrics(predictions, labels):
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("invalid MOSI evaluation arrays")
    pp, yy = (p >= 0), (y >= 0)
    return {"mae": float(np.mean(np.abs(p-y))), "mse": float(np.mean((p-y)**2)),
            "corr": _corr(p,y), "acc2": float(np.mean(pp == yy)),
            "f1": _weighted_f1(yy,pp),
            "acc7": float(np.mean(np.clip(np.rint(p),-3,3) == np.clip(np.rint(y),-3,3)))}


def simsv2_metrics(predictions, labels):
    p = np.clip(np.asarray(predictions, dtype=np.float64).reshape(-1),-1,1)
    y = np.clip(np.asarray(labels, dtype=np.float64).reshape(-1),-1,1)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("invalid SIMSv2 evaluation arrays")
    # Original CH-SIMSv2 script uses interval (left,right], i.e. 0 -> negative.
    def bin_vals(values, boundaries):
        return np.digitize(values, np.asarray(boundaries[1:-1]), right=True)
    p2,y2 = bin_vals(p,[-1.01,0,1.01]), bin_vals(y,[-1.01,0,1.01])
    return {"mae":float(np.mean(np.abs(p-y))),"mse":float(np.mean((p-y)**2)),
            "corr":_corr(p,y),
            "acc2_percent":float(np.mean(p2 == y2)*100),"f1_percent":_weighted_f1(y2,p2)*100,
            "acc3_percent":float(np.mean(bin_vals(p,[-1.01,-.1,.1,1.01]) == bin_vals(y,[-1.01,-.1,.1,1.01]))*100),
            "acc5_percent":float(np.mean(bin_vals(p,[-1.01,-.7,-.1,.1,.7,1.01]) == bin_vals(y,[-1.01,-.7,-.1,.1,.7,1.01]))*100)}
