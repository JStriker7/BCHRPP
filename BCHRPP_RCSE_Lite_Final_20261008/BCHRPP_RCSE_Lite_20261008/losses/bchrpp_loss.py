"""Compact training objective: task + unimodal probes + quality calibration."""
import torch
import torch.nn.functional as F


def task_loss(prediction, label, task='regression', sample_weights=None):
    if task == 'classification':
        losses = F.cross_entropy(prediction, label.long().view(-1), reduction='none')
    elif task == 'regression':
        losses = (0.7 * F.smooth_l1_loss(prediction, label, beta=.5, reduction='none')
                  + 0.3 * (prediction - label).square()).mean(-1)
    else:
        raise ValueError(f'unsupported task: {task}')
    if sample_weights is not None:
        w = sample_weights.reshape(-1).to(losses)
        return (losses*w).sum() / w.sum().clamp_min(1e-6)
    return losses.mean()


def compute_loss(prediction, label, aux, task='regression', sample_weights=None,
                 aux_weight=.10, confidence_weight=.05):
    main = task_loss(prediction, label, task, sample_weights)
    present = aux['modality_available'].to(prediction.dtype)
    if sample_weights is not None:
        present = present * sample_weights.reshape(-1, 1).to(prediction.dtype)
    preds = aux['unimodal_predictions']
    b, m, c = preds.shape
    if task == 'classification':
        expanded = label.long().reshape(-1, 1).expand(-1, m).reshape(-1)
        per_modality = F.cross_entropy(preds.reshape(b*m,c), expanded, reduction='none').reshape(b,m)
        # For classification, p(true class) is a scale-free quality proxy.
        quality = preds.detach().softmax(-1).gather(-1, label.long().reshape(b,1,1).expand(-1,m,1)).squeeze(-1)
    else:
        target = label.reshape(b, 1, c).expand_as(preds)
        per_modality = F.smooth_l1_loss(preds, target, beta=.5, reduction='none').mean(-1)
        quality = torch.exp(-(preds.detach()-target).abs().mean(-1))
    den = present.sum().clamp_min(1.)
    uni = (per_modality * present).sum() / den
    confidence = aux['confidence']
    # Quality is explicitly detached: cannot reduce calibration loss by
    # deliberately making the modality probe worse.
    calibration = ((confidence-quality).square() * present).sum() / den
    total = main + aux_weight*uni + confidence_weight*calibration
    return total, {'main':main.detach(), 'unimodal':uni.detach(),
                   'calibration':calibration.detach(), 'total':total.detach()}
