
import numpy as np

def mae(y,p):
    return np.mean(abs(y-p))

def corr(y,p):
    return np.corrcoef(y,p)[0,1]

def acc(y,p):
    return (y==p).mean()


def _binary_scores(truth, prediction):
    if len(truth) == 0:
        return None, None
    accuracy = float(np.mean(truth == prediction))
    weighted_f1 = 0.0
    for label in (False, True):
        support = int(np.sum(truth == label))
        true_positive = int(np.sum((truth == label) & (prediction == label)))
        false_positive = int(np.sum((truth != label) & (prediction == label)))
        false_negative = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        weighted_f1 += support * (2 * true_positive / denominator if denominator else 0.0)
    return accuracy, float(weighted_f1 / len(truth))


def mosi_metrics(truth, prediction):
    """MOSI regression protocol: raw MAE; clipped/rounded [-3,3] Acc-7."""
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if len(truth) == 0 or truth.shape != prediction.shape:
        raise ValueError("MOSI predictions and labels must have matching nonempty shapes")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("MOSI predictions and labels must be finite")
    nonzero = truth != 0
    acc2_nz, f1_nz = _binary_scores(truth[nonzero] > 0, prediction[nonzero] > 0)
    acc2_all, f1_all = _binary_scores(truth >= 0, prediction >= 0)
    correlation = 0.0
    if len(truth) > 1 and np.std(truth) > 0 and np.std(prediction) > 0:
        correlation = float(np.corrcoef(truth, prediction)[0, 1])
    return {
        "Acc-7": float(np.mean(np.rint(np.clip(truth, -3, 3)) ==
                               np.rint(np.clip(prediction, -3, 3)))),
        "MAE": float(np.mean(np.abs(truth - prediction))),
        "Corr": correlation,
        "Acc-2_nonzero": acc2_nz,
        "F1_nonzero": f1_nz,
        "Acc-2_haszero": acc2_all,
        "F1_haszero": f1_all,
    }
