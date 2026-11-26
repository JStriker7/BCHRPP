"""BCHR++ RCSE-Lite: reliability-complementarity coupled sparse evidence reasoning.

Three architectural steps: (1) mask-safe local encoding/query pooling,
(2) ONE reliability-complementarity router shared by fast aggregation and
    sparse message passing,
(3) residual structured correction of the fast prediction.

This implementation does not claim identified causality, true hyperedges,
Mamba operations, or benchmark SOTA results. See docs/METHOD_AND_PROTOCOL.md.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

MODALITIES = ('text', 'audio', 'vision')


def masked_softmax(logits, mask, dim=-1):
    """Masked softmax; completely masked rows map to zero (not NaN)."""
    mask = mask.to(dtype=torch.bool, device=logits.device)
    scores = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(scores, dim=dim) * mask.to(dtype=logits.dtype)
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-9)


class EvidenceEncoder(nn.Module):
    """Shared learnable query tokens, optional depthwise local filter."""

    def __init__(self, d=128, q=4, temporal=True, dropout=0.1):
        super().__init__()
        if q < 1:
            raise ValueError('q must be positive')
        self.query = nn.Parameter(torch.randn(q, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.temporal_enabled = temporal
        if temporal:
            self.local = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
            nn.init.normal_(self.local.weight, std=0.01)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError('evidence must have shape [B,T,D] and T > 0')
        b, t, d = x.shape
        if mask is None:
            mask = torch.ones(b, t, device=x.device, dtype=torch.bool)
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)
        if mask.shape != (b, t):
            raise ValueError('mask must have shape [B,T]')
        valid = mask.unsqueeze(-1)
        x = self.norm(x * valid) * valid
        if self.temporal_enabled:
            x = x + self.drop(self.local(x.transpose(1, 2)).transpose(1, 2)) * valid
        scores = torch.einsum('qd,btd->bqt', self.query, x) / math.sqrt(d)
        attention = masked_softmax(scores, mask.unsqueeze(1), dim=-1)
        return torch.bmm(attention, x)


class CoupledEvidenceRouter(nn.Module):
    """One distribution for fast consensus AND graph key authority.

    r_m = sigmoid(g(h_m)) predicts per-modality outcome quality.
    c_m = r_m * weighted_{j != m}(r_j * (1 - cos(h_m,h_j))/2).
    pi_m = softmax_masked(log r_m + gamma * c_m), gamma in [0,1].

    Complementarity is discounted by its own reliability and the reliability
    of peers: a dissimilar but untrusted feature cannot dominate by default.
    No causal meaning is attached to disagreement/complementarity.
    """

    def __init__(self, d, reliability=True, complementarity=True, temperature=1.):
        super().__init__()
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.temperature = float(temperature)
        self.use_reliability = reliability
        self.use_complementarity = complementarity
        self.quality_head = nn.Linear(d, 1)
        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)
        self.complementarity_logit = nn.Parameter(torch.tensor(-1.38629436))  # sigmoid -> 0.2

    def forward(self, reps, available):
        b, m, d = reps.shape
        confidence = torch.sigmoid(self.quality_head(reps).squeeze(-1))
        if not self.use_reliability:
            confidence = torch.ones_like(confidence)
        confidence = confidence * available
        unit = F.normalize(reps, p=2, dim=-1, eps=1e-6)
        cosine = torch.bmm(unit, unit.transpose(1, 2))
        dissimilarity = (1. - cosine).clamp(0, 2) * 0.5
        other = (~torch.eye(m, device=reps.device, dtype=torch.bool)).unsqueeze(0)
        partner = confidence.unsqueeze(1) * available.unsqueeze(1) * other
        complement = (dissimilarity * partner).sum(-1) / partner.sum(-1).clamp_min(1e-6)
        complement = complement * confidence
        if not self.use_complementarity:
            complement = torch.zeros_like(complement)
        gamma = torch.sigmoid(self.complementarity_logit)
        logits = torch.log(confidence.clamp_min(1e-6)) + gamma * complement
        weights = masked_softmax(logits / self.temperature, available, dim=-1)
        return weights, confidence, complement


class SparseEvidenceReasoner(nn.Module):
    """ONE sparse association layer. No second Transformer stack.

    Top-k is per evidence query. Invalid keys and query nodes never contribute.
    Includes self-neighbors as a safe fall-back, followed by gated residual.
    """

    def __init__(self, d, topk=4, dropout=0.1, sparse_graph=True):
        super().__init__()
        if topk < 0:
            raise ValueError('topk cannot be negative')
        self.topk = int(topk)
        self.sparse_graph = sparse_graph
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(2*d, d)
        nn.init.constant_(self.gate.bias, -2.)
        self.out = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_mask, authority):
        b, n, d = x.shape
        if not node_mask.any(dim=1).all():
            raise ValueError('at least one valid evidence node is required')
        q, k = F.normalize(self.q(x), dim=-1), F.normalize(self.k(x), dim=-1)
        similarity = torch.bmm(q, k.transpose(1, 2)) * math.sqrt(d)
        similarity = similarity + torch.log(authority.clamp_min(1e-6)).unsqueeze(1)
        valid_keys = node_mask.unsqueeze(1).expand(-1, n, -1)
        if self.sparse_graph and self.topk and self.topk < n:
            # Mask FIRST, top-k SECOND: no padded key can enter a selected edge.
            select_score = similarity.masked_fill(~valid_keys, torch.finfo(similarity.dtype).min)
            indices = select_score.topk(min(self.topk, n), dim=-1).indices
            chosen = torch.zeros_like(valid_keys).scatter(-1, indices, True)
            valid_keys = valid_keys & chosen
        attn = masked_softmax(similarity, valid_keys, dim=-1)
        message = self.out(torch.bmm(attn, self.v(x)))
        gating = torch.sigmoid(self.gate(torch.cat((x, message), dim=-1)))
        result = self.norm(x + gating * self.dropout(message))
        return result * node_mask.unsqueeze(-1)


class ResidualArbiter(nn.Module):
    """Direct prediction + bounded structural correction, not two independent heads."""

    def __init__(self, d, output_dim=1):
        super().__init__()
        self.direct = nn.Linear(d, output_dim)
        self.delta = nn.Linear(d, output_dim, bias=False)
        nn.init.normal_(self.delta.weight, std=0.01)
        self.gate = nn.Linear(d + 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.)

    def forward(self, direct_pool, relation_pool, avg_confidence):
        fast = self.direct(direct_pool)
        refinement = relation_pool - direct_pool
        use_relations = torch.sigmoid(self.gate(torch.cat((refinement, avg_confidence), dim=-1)))
        prediction = fast + use_relations * self.delta(refinement)
        return prediction, fast, use_relations


class BCHRPP(nn.Module):
    """RCSE-Lite for three-modal sentiment regression / logits classification."""

    def __init__(self, d=128, output_dim=1, queries=4,
                 text_dim=768, audio_dim=5, vision_dim=20, dropout=0.1,
                 topk=4, temporal=True, sparse_graph=True,
                 reliability=True, complementarity=True, refinement=True,
                 modality_dropout=0.1, routing_temperature=1.):
        super().__init__()
        if d < 1 or queries < 1 or output_dim < 1:
            raise ValueError('d, queries and output_dim must be positive')
        if not 0 <= modality_dropout < 1:
            raise ValueError('modality_dropout must lie in [0,1)')
        self.hidden_dim = d
        self.output_dim = output_dim
        self.queries = queries
        self.modality_dropout = modality_dropout
        self.use_refinement = refinement
        self.text_projection = nn.Linear(text_dim, d)
        self.audio_projection = nn.Linear(audio_dim, d)
        self.vision_projection = nn.Linear(vision_dim, d)
        self.modality_embedding = nn.Parameter(torch.randn(3, d) * 0.01)
        self.dropout = nn.Dropout(dropout)
        self.evidence_encoder = EvidenceEncoder(d, queries, temporal=temporal, dropout=dropout)
        self.probe = nn.Linear(d, output_dim)  # ONE shared auxiliary head
        self.router = CoupledEvidenceRouter(d, reliability, complementarity, routing_temperature)
        self.reasoner = SparseEvidenceReasoner(d, topk, dropout, sparse_graph)
        self.arbiter = ResidualArbiter(d, output_dim)

    def forward(self, text, audio, vision, masks=None, return_aux=False):
        masks = masks or {}
        inputs = [text, audio, vision]
        projects = [self.text_projection, self.audio_projection, self.vision_projection]
        encoded, present = [], []
        for idx, (name, x, project) in enumerate(zip(MODALITIES, inputs, projects)):
            if x.ndim != 3:
                raise ValueError(f'{name} must have shape [B,T,D]')
            if x.shape[-1] != project.in_features:
                raise ValueError(f'{name} input feature dimension mismatch')
            valid = masks.get(name)
            if valid is None:
                valid = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
            else:
                valid = valid.to(x.device, dtype=torch.bool)
            if valid.shape != x.shape[:2]:
                raise ValueError(f'{name} mask shape mismatch')
            z = self.dropout(project(x * valid.unsqueeze(-1)))
            # Adding a modality identity is done before masking in EvidenceEncoder;
            # an absent modality still gets exact zero tokens.
            z = z + self.modality_embedding[idx]
            encoded.append(self.evidence_encoder(z, valid))
            present.append(valid.any(dim=-1))
        available = torch.stack(present, dim=1)
        if not available.any(dim=1).all():
            raise ValueError('at least one modality per sample must be present')
        if self.training and self.modality_dropout:
            dropped = ((torch.rand_like(available, dtype=torch.float) < self.modality_dropout)
                       & available)
            empty = (available & ~dropped).sum(dim=1) == 0
            first = available.to(torch.int64).argmax(dim=1)
            dropped[empty, first[empty]] = False
            available = available & ~dropped
        tokens = torch.stack(encoded, dim=1)  # [B,3,Q,D]
        tokens = tokens * available[:, :, None, None]
        reps = tokens.mean(dim=2)
        modality_prediction = self.probe(reps)
        weights, confidence, complementarity = self.router(reps, available)
        fast_pool = torch.sum(reps * weights.unsqueeze(-1), dim=1)
        node_mask = available.repeat_interleave(self.queries, dim=1)
        node_authority = weights.repeat_interleave(self.queries, dim=1)
        flat_nodes = tokens.flatten(1, 2)
        if self.use_refinement:
            refined = self.reasoner(flat_nodes, node_mask, node_authority)
            # IMPORTANT: use SAME routing distribution in relation readout.
            relation_pool = torch.sum(refined * node_authority.unsqueeze(-1), dim=1)
            # Each modality has Q tokens, thus node_authority sums to Q.
            relation_pool = relation_pool / self.queries
        else:
            relation_pool = fast_pool
        avg_conf = (confidence * weights).sum(1, keepdim=True)
        output, direct, rel_gate = self.arbiter(fast_pool, relation_pool, avg_conf)
        if not return_aux:
            return output
        return output, {
            'unimodal_predictions': modality_prediction,
            'confidence': confidence, 'modality_weights': weights,
            'modality_available': available,
            'complementarity': complementarity,
            'modality_representations': reps,
            'direct_prediction': direct, 'refinement_gate': rel_gate,
        }
