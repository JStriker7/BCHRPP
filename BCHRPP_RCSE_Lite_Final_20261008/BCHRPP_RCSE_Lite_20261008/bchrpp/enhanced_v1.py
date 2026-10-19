"""BCHR++ enhanced: lightweight evidence routing, graph reasoning, and arbitration.

Reference-inspired *adaptations* (not reproductions of published models):
CICA reliability calibration; KGCF-GA loss gradient alignment; GraphMamba
local temporal filtering / sparse structural reasoning; MSBF dynamic dominance;
CASDF missing-modality-safe association; CAMS-AMSS noise-aware routing.
No external SenticNet/ConceptNet resource or Mamba dependency is assumed.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


MODALITIES = ("text", "audio", "vision")


def masked_mean(x, mask, dim=1):
    w = mask.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.)


class EvidenceEncoder(nn.Module):
    """Masked learned-query compression with optional local temporal filtering."""

    def __init__(self, d=256, q=4, temporal=True, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(q, d) * 0.02)
        self.input_norm = nn.LayerNorm(d)
        self.temporal_enabled = temporal
        self.temporal = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d)
        nn.init.normal_(self.temporal.weight, std=0.01)
        nn.init.zeros_(self.temporal.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if x.ndim != 3:
            raise ValueError("evidence inputs must be [B,T,D]")
        b, length, _ = x.shape
        if length == 0:
            raise ValueError("sequence length must be positive")
        if mask is None:
            mask = torch.ones(b, length, dtype=torch.bool, device=x.device)
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.shape != x.shape[:2]:
                raise ValueError("mask must be [batch,sequence_length]")
        # Crucial: remove padded values BEFORE the convolution, not only at pooling.
        x = self.input_norm(x * mask.unsqueeze(-1))
        x = x * mask.unsqueeze(-1)
        if self.temporal_enabled:
            x = x + self.dropout(self.temporal(x.transpose(1, 2)).transpose(1, 2)) * mask.unsqueeze(-1)
        available = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_mask[~available, 0] = True
        scores = torch.einsum("qd,btd->bqt", self.query, x) / math.sqrt(x.size(-1))
        scores = scores.masked_fill(~safe_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        attn = scores.softmax(dim=-1)
        tokens = attn @ x
        return tokens * available[:, None, None].to(tokens.dtype)


class CausalDisentangler(nn.Module):
    """Task-evidence, contextual, nuisance decomposition with reconstruction.

    Branch names retain the source API; these projections are NOT proof of
    identifiable causality. Orthogonality/reconstruction help avoid collapse.
    """

    def __init__(self, d):
        super().__init__()
        self.causal = nn.Linear(d, d)
        self.appearance = nn.Linear(d, d)
        self.nuisance = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.reconstruct = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        nn.init.constant_(self.gate.bias, 0.)

    def forward(self, x):
        c = self.norm(x + torch.sigmoid(self.gate(x)) * torch.tanh(self.causal(x)))
        a = torch.tanh(self.appearance(x))
        n = torch.tanh(self.nuisance(x))
        reconstructed = self.reconstruct((c + a + n) / 3.)
        return c, a, n, reconstructed


class CounterfactualContribution(nn.Module):
    """Leave-one-modality-out impact on the FAST prediction, no extra backbone.

    These are predictive perturbations, NOT causal treatment-effect estimates.
    """

    def forward(self, reps, available, fast_head, weights):
        # reps [B,M,D]; fast_head: pooled representation -> logits
        full = (reps * weights.unsqueeze(-1)).sum(dim=1)
        base_pred = fast_head(full)
        impacts = []
        for k in range(reps.size(1)):
            leave = weights.clone()
            leave[:, k] = 0
            leave = leave / leave.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            # No candidate exists if all available sources were removed.
            other = (reps * leave.unsqueeze(-1)).sum(dim=1)
            delta = (base_pred - fast_head(other)).abs().mean(dim=-1)
            impacts.append(delta)
        return torch.stack(impacts, dim=1) * available.to(reps.dtype)


class CognitiveCompetition(nn.Module):
    """Reliability-informed differentiable primary-modality competition."""

    def __init__(self, temperature=1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("competition temperature must be positive")
        self.temperature = temperature

    def forward(self, confidence, available, cf_importance=None):
        scores = torch.log(confidence.clamp_min(1e-5))
        if cf_importance is not None:
            # Small bounded preference for unique predictive evidence.
            cf_norm = cf_importance / (cf_importance.sum(dim=1, keepdim=True) + 1e-6)
            scores = scores + 0.25 * cf_norm.detach()
        scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
        return torch.softmax(scores / self.temperature, dim=-1)


class CausalHypergraphTransformer(nn.Module):
    """Sparse similarity graph + slot-level cross-modal incidence + Transformer.

    The learned graph is associative: without interventions or a causal graph,
    its edges should not be interpreted as proven causal relationships.
    """

    def __init__(self, d, heads=4, layers=2, topk=5, dropout=0.1, sparse_graph=True):
        super().__init__()
        if d % heads != 0:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.topk = topk
        self.sparse_graph = sparse_graph
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d)
        self.node_gate = nn.Linear(d * 2, d)
        self.slot_gate = nn.Linear(d * 2, d)
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=2*d,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        nn.init.constant_(self.node_gate.bias, -2.5)
        nn.init.constant_(self.slot_gate.bias, -2.5)

    def forward(self, x, mask=None, source_weights=None):
        b, n, d = x.shape
        if mask is None:
            mask = torch.ones(b, n, device=x.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("each sample requires at least one valid evidence token")
        if source_weights is None:
            source_weights = torch.ones(b, n, device=x.device, dtype=x.dtype)
        q = F.normalize(self.q_proj(x), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj(x), dim=-1, eps=1e-6)
        score = torch.bmm(q, k.transpose(1, 2)) * math.sqrt(d)
        score = score + torch.log(source_weights.clamp_min(1e-6))[:, None, :]
        score = score.masked_fill(~mask[:, None, :], torch.finfo(score.dtype).min)
        if self.sparse_graph and self.topk and self.topk < n:
            # Exclude padded keys from sparse selections; self loops remain eligible.
            indices = torch.topk(score, min(self.topk, n), dim=-1).indices
            adjacency = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, indices, True)
            score = score.masked_fill(~adjacency, torch.finfo(score.dtype).min)
        graph_msg = torch.bmm(score.softmax(dim=-1), self.v_proj(x))
        gate = torch.sigmoid(self.node_gate(torch.cat([x, graph_msg], dim=-1)))
        x = x + gate * graph_msg
        # Query slots are matched across T/A/V; availability/reliability normalizes
        # incidence weights so that missing modalities do not create phantom edges.
        if n % 3 == 0:
            query_count = n // 3
            grouped = x.reshape(b, 3, query_count, d)
            w = source_weights.reshape(b, 3, query_count) * mask.reshape(b, 3, query_count)
            slot_context = (grouped * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)
            shared = slot_context.unsqueeze(1).expand_as(grouped).reshape(b, n, d)
            x = x + torch.sigmoid(self.slot_gate(torch.cat([x, shared], dim=-1))) * shared
        x = self.norm(x)
        x = x * mask.unsqueeze(-1)
        x = self.encoder(x, src_key_padding_mask=~mask)
        return x * mask.unsqueeze(-1)


class DualArbitration(nn.Module):
    """Fast unimodal consensus and slow relational evidence with disagreement gate."""

    def __init__(self, d, output_dim):
        super().__init__()
        self.fast = nn.Linear(d, output_dim)
        self.slow = nn.Linear(d, output_dim)
        self.gate = nn.Linear(d, 1)
        self.conflict_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.constant_(self.gate.bias, -0.3)

    def forward(self, raw_pool, reasoned, evidence_mask, modality_reps, modality_weights):
        slow_pool = masked_mean(reasoned, evidence_mask)
        fast_pred = self.fast(raw_pool)
        slow_pred = self.slow(slow_pool)
        normed = F.normalize(modality_reps, dim=-1, eps=1e-6)
        similarities = normed @ normed.transpose(1, 2)
        pair_weight = modality_weights[:, :, None] * modality_weights[:, None, :]
        nonself = 1. - torch.eye(3, device=reasoned.device, dtype=reasoned.dtype)
        disagree = ((1. - similarities).clamp_min(0.) * nonself * pair_weight).sum((1, 2))
        denom = (nonself * pair_weight).sum((1, 2)).clamp_min(1e-6)
        conflict = (disagree / denom).unsqueeze(-1)
        slow_gate = torch.sigmoid(self.gate(slow_pool - raw_pool) + F.softplus(self.conflict_scale) * conflict)
        prediction = (1. - slow_gate) * fast_pred + slow_gate * slow_pred
        return prediction, fast_pred, slow_pred, slow_gate


class BCHRPP(nn.Module):
    """Enhanced BCHR++ multimodal regression/classification with optional aux.

    Default input dimensions are the original MOSI features. Pass dimensions
    explicitly or let the training script infer them from the dataset.
    """

    def __init__(self, d=256, output_dim=1, queries=4, text_dim=768, audio_dim=5,
                 vision_dim=20, dropout=0.1, topk=5, temporal=True,
                 sparse_graph=True, reliability=True, counterfactual=True,
                 modality_dropout=0.10, competition_temperature=1.0):
        super().__init__()
        self.hidden_dim, self.queries, self.output_dim = d, queries, output_dim
        self.use_reliability = reliability
        self.use_counterfactual = counterfactual
        self.modality_dropout = modality_dropout
        self.text_projection = nn.Linear(text_dim, d)
        self.audio_projection = nn.Linear(audio_dim, d)
        self.vision_projection = nn.Linear(vision_dim, d)
        self.projection_dropout = nn.Dropout(dropout)
        self.evidence_encoder = EvidenceEncoder(d, queries, temporal=temporal, dropout=dropout)
        self.disentangler = CausalDisentangler(d)
        self.counterfactual = CounterfactualContribution()
        self.competition = CognitiveCompetition(competition_temperature)
        self.reasoner = CausalHypergraphTransformer(d, topk=topk, dropout=dropout, sparse_graph=sparse_graph)
        self.head = DualArbitration(d, output_dim)
        self.unimodal_heads = nn.ModuleDict({m: nn.Linear(d, output_dim) for m in MODALITIES})
        self.confidence_heads = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, 1)) for m in MODALITIES})
        # Near-uniform initial routing; reliability learns from auxiliary signals.
        for branch in self.confidence_heads.values():
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def encode_modality(self, values, projection, mask=None):
        if values.ndim != 3:
            raise ValueError("modality input must be [B,T,D]")
        if mask is None:
            mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        else:
            mask = mask.to(device=values.device, dtype=torch.bool)
            if mask.shape != values.shape[:2]:
                raise ValueError("modality mask dimension mismatch")
        # Remove padding before linear bias and local temporal mixing.
        projected = self.projection_dropout(projection(values * mask.unsqueeze(-1)))
        evidence = self.evidence_encoder(projected, mask)
        causal, appearance, nuisance, reconstruction = self.disentangler(evidence)
        availability = mask.any(dim=1, keepdim=True)
        evidence_mask = availability.expand(-1, causal.size(1))
        return (causal * evidence_mask.unsqueeze(-1),
                appearance * evidence_mask.unsqueeze(-1),
                nuisance * evidence_mask.unsqueeze(-1),
                reconstruction * evidence_mask.unsqueeze(-1),
                evidence, evidence_mask)

    def forward(self, text, audio, vision, masks=None, return_aux=False):
        masks = masks or {}
        inputs = (text, audio, vision)
        projections = (self.text_projection, self.audio_projection, self.vision_projection)
        encoded = [self.encode_modality(x, p, masks.get(m)) for m, x, p in zip(MODALITIES, inputs, projections)]
        available = torch.stack([e[-1].any(dim=1) for e in encoded], dim=1)
        if not available.any(dim=1).all():
            raise ValueError("at least one modality per sample must be present")
        if self.training and self.modality_dropout > 0:
            drops = (torch.rand(available.shape, device=available.device) < self.modality_dropout) & available
            all_dropped = (available & ~drops).sum(dim=1) == 0
            # Retain the first available modality when stochastic masking removes all.
            first_available = available.to(torch.int64).argmax(dim=1)
            drops[all_dropped, first_available[all_dropped]] = False
            available = available & ~drops
        else:
            drops = torch.zeros_like(available)
        mask = torch.cat([e[-1] & available[:, i:i+1] for i, e in enumerate(encoded)], dim=1)
        source_repr = torch.stack([
            masked_mean(e[0], e[-1]) * available[:, i:i+1]
            for i, e in enumerate(encoded)
        ], dim=1)
        unipred = torch.stack([self.unimodal_heads[m](source_repr[:, i]) for i,m in enumerate(MODALITIES)], dim=1)
        conf = torch.stack([self.confidence_heads[m](source_repr[:, i]).squeeze(-1).sigmoid() for i,m in enumerate(MODALITIES)], dim=1)
        if not self.use_reliability:
            conf = torch.ones_like(conf)
        weights = self.competition(conf, available)
        cf_impact = None
        if self.use_counterfactual:
            cf_impact = self.counterfactual(source_repr, available, self.head.fast, weights)
            weights = self.competition(conf, available, cf_impact)
        fast_pool = (weights.unsqueeze(-1) * source_repr).sum(dim=1)
        nodes = torch.cat([e[0] for e in encoded], dim=1)
        nodes = nodes * mask.unsqueeze(-1)
        node_weights = weights.repeat_interleave(self.queries, dim=1)
        reasoned = self.reasoner(nodes, mask=mask, source_weights=node_weights)
        pred, fast_pred, slow_pred, slow_gate = self.head(fast_pool, reasoned, mask, source_repr, weights)
        if not return_aux:
            return pred
        factors = {
            "causal": torch.stack([e[0] for e in encoded], dim=1),
            "appearance": torch.stack([e[1] for e in encoded], dim=1),
            "nuisance": torch.stack([e[2] for e in encoded], dim=1),
            "reconstructed": torch.stack([e[3] for e in encoded], dim=1),
            "input_evidence": torch.stack([e[4] for e in encoded], dim=1),
        }
        return pred, {
            "unimodal_predictions": unipred, "confidence": conf,
            "modality_weights": weights, "modality_available": available,
            "counterfactual_importance": cf_impact, "factors": factors,
            "modality_representations": source_repr,
            "fast_prediction": fast_pred, "slow_prediction": slow_pred,
            "slow_gate": slow_gate,
        }
