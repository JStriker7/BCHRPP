"""Mask-aware BCHR++ objectives, including optional gradient alignment.

The objectives regularize predictive factors, not identifiable causal variables.
"""
import torch
import torch.nn.functional as F


def factor_orthogonal_loss(a, b, mask=None):
    # Cosine decorrelation on normalized token vectors; bounded with batch size.
    value = (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1).square()
    if mask is None:
        return value.mean()
    w = mask.to(value.dtype).unsqueeze(-1)
    return (value * w).sum() / (w.sum() * value.shape[-1]).clamp_min(1.)


def reconstruction_loss(x, xhat, mask=None):
    value = (x - xhat).square().mean(dim=-1)
    if mask is None:
        return value.mean()
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum() / (weight.sum() * value.shape[-1]).clamp_min(1.)


def task_loss(prediction, label, task="regression", sample_weights=None):
    if task == "classification":
        individual = F.cross_entropy(prediction, label.long().view(-1), reduction="none")
    elif task == "regression":
        individual = (0.7 * F.smooth_l1_loss(prediction, label, beta=0.5, reduction="none")
                      + 0.3 * (prediction-label).square()).mean(dim=-1)
    else:
        raise ValueError(f"unknown task: {task}")
    if sample_weights is not None:
        individual = individual * sample_weights.flatten().to(individual.dtype)
    return individual.mean()


def _aux_loss(pred, label, mask, task, batch_weight=None):
    b, m, c = pred.shape
    valid = mask.to(pred.dtype)
    if task == "classification":
        error = F.cross_entropy(pred.reshape(b*m, c), label.long().view(-1, 1).expand(-1,m).reshape(-1), reduction="none").view(b,m)
        calibration_target = (-error.detach()).exp().clamp(0., 1.)
    else:
        error = F.smooth_l1_loss(pred, label.unsqueeze(1).expand_as(pred), beta=0.5, reduction="none").mean(dim=-1)
        calibration_target = torch.exp(-(pred.detach() - label.unsqueeze(1)).abs().mean(-1))
    if batch_weight is not None:
        valid = valid * batch_weight[:, None]
    return (error * valid).sum() / valid.sum().clamp_min(1.), calibration_target


def gradient_alignment_weight(task_term, aux_term, shared_representations, valid, enabled=True):
    """KGCF-GA-inspired cosine scaling of auxiliary loss, no higher-order graph."""
    if not enabled or not shared_representations.requires_grad:
        return task_term.new_tensor(1.)
    gt = torch.autograd.grad(task_term, shared_representations, retain_graph=True, allow_unused=True)[0]
    ga = torch.autograd.grad(aux_term, shared_representations, retain_graph=True, allow_unused=True)[0]
    if gt is None or ga is None:
        return task_term.new_tensor(1.)
    gt, ga = gt.detach(), ga.detach()
    cos = F.cosine_similarity(gt, ga, dim=-1, eps=1e-6)
    v = valid.to(cos.dtype)
    mean_cos = (cos * v).sum() / v.sum().clamp_min(1.)
    # Do not zero out aux task entirely under conflict; bounded [0.5, 1.5].
    return (1. + 0.5 * mean_cos).clamp(0.5, 1.5).detach()


def compute_loss(prediction, label, aux, task="regression", sample_weights=None,
                 aux_weight=0.15, confidence_weight=0.05, factor_weight=0.01,
                 gradient_alignment=True):
    main = task_loss(prediction, label, task, sample_weights)
    valid = aux["modality_available"]
    uni = aux["unimodal_predictions"]
    unimodal, target_quality = _aux_loss(uni, label, valid, task, sample_weights)
    if aux_weight > 0 and gradient_alignment:
        scale = gradient_alignment_weight(main, unimodal, aux["modality_representations"], valid)
    else:
        scale = main.new_tensor(1.)
    confidence = aux["confidence"]
    if confidence.requires_grad:
        squared = (confidence - target_quality).square() * valid
        calibration = squared.sum() / valid.sum().clamp_min(1.)
    else:
        calibration = main.new_tensor(0.)
    f = aux["factors"]
    # Modality dropout should not regularize hidden signals from dropped sources.
    reconstruction = reconstruction_loss(f["input_evidence"], f["reconstructed"], valid)
    orth = (factor_orthogonal_loss(f["causal"], f["appearance"], valid)
            + factor_orthogonal_loss(f["causal"], f["nuisance"], valid)) * 0.5
    factor = reconstruction + 0.1 * orth
    total = main + aux_weight * scale * unimodal + confidence_weight * calibration + factor_weight * factor
    terms = {"main": main.detach(), "unimodal": unimodal.detach(),
             "gradient_alignment_scale": scale.detach(), "calibration": calibration.detach(),
             "factor": factor.detach(), "total": total.detach()}
    return total, terms


# Keep the source convenience API functional.
def total_loss(task, factor, cf, hyper, l1=1, l2=1, l3=1):
    return task + l1*factor + l2*cf + l3*hyper


def counterfactual_loss(p, p_cf):
    # Positive numerical safeguard and symmetric divergence for probability vectors.
    p = p.clamp_min(1e-8)
    p_cf = p_cf.clamp_min(1e-8)
    mix = (p + p_cf) * .5
    return 0.5 * ((p * (p.log() - mix.log())).sum(-1).mean()
                  + (p_cf * (p_cf.log()-mix.log())).sum(-1).mean())
