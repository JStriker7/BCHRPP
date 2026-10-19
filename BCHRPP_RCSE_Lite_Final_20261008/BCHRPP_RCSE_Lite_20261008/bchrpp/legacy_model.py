import math

import torch
import torch.nn as nn


class EvidenceEncoder(nn.Module):
    """Compress one modality sequence into evidence tokens."""

    def __init__(self, d=256, q=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(q, d))

    def forward(self, x, mask=None):
        score = torch.matmul(
            self.query,
            x.transpose(1, 2),
        ) / math.sqrt(x.size(-1))

        if mask is not None:
            mask = mask.to(
                dtype=torch.bool,
                device=x.device,
            )

            if mask.ndim != 2:
                raise ValueError(
                    "mask must have shape "
                    "[batch, sequence_length]"
                )

            if mask.shape != x.shape[:2]:
                raise ValueError(
                    "mask shape does not match input shape"
                )

            # Some MOSI samples may have an entire modality missing.
            # Temporarily expose the first position so softmax does not
            # receive a row containing only negative infinity.
            available = mask.any(dim=1)
            safe_mask = mask.clone()
            safe_mask[~available, 0] = True

            score = score.masked_fill(
                ~safe_mask[:, None, :],
                torch.finfo(score.dtype).min,
            )

        attention = torch.softmax(score, dim=-1)
        evidence = torch.matmul(attention, x)

        return evidence


class CausalDisentangler(nn.Module):
    """
    Split evidence into causal, appearance and nuisance factors.

    This is currently a simple three-branch projection. The complete
    reconstruction and disentanglement losses are added in later stages.
    """

    def __init__(self, d):
        super().__init__()

        self.causal = nn.Linear(d, d)
        self.appearance = nn.Linear(d, d)
        self.nuisance = nn.Linear(d, d)

    def forward(self, x):
        causal = self.causal(x)
        appearance = self.appearance(x)
        nuisance = self.nuisance(x)

        return causal, appearance, nuisance


class CounterfactualContribution(nn.Module):
    """
    Temporary counterfactual contribution calculation.

    The complete training-time CCE described in the method document
    has not yet been implemented.
    """

    def forward(self, full, masked):
        return torch.norm(
            full - masked,
            dim=-1,
        )


class CognitiveCompetition(nn.Module):
    """
    Temporary cognitive competition calculation.

    The complete fixed-point competition described in the method
    document has not yet been implemented.
    """

    def forward(self, energy):
        return torch.softmax(
            energy,
            dim=-1,
        )


class CausalHypergraphTransformer(nn.Module):
    """
    Temporary reasoning backbone.

    This is currently a standard Transformer encoder. It is not yet
    the complete self-induced causal hypergraph Transformer described
    in the BCHR++ method document.
    """

    def __init__(self, d):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=4,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
        )

    def forward(self, x, mask=None):
        padding_mask = None

        if mask is not None:
            mask = mask.to(
                dtype=torch.bool,
                device=x.device,
            )

            if mask.shape != x.shape[:2]:
                raise ValueError(
                    "evidence mask shape does not match "
                    "evidence tensor"
                )

            # Transformer uses True for positions that must be ignored.
            padding_mask = ~mask

        return self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )


class DualArbitration(nn.Module):
    """
    Temporary fast and slow prediction head.

    At this stage both paths use the same pooled representation.
    The complete conflict-triggered dual-process design will be
    implemented later.
    """

    def __init__(self, d, output_dim):
        super().__init__()

        self.fast = nn.Linear(d, output_dim)
        self.slow = nn.Linear(d, output_dim)
        self.gate = nn.Linear(d, 1)

    def forward(self, x, mask=None):
        if mask is None:
            hidden = x.mean(dim=1)
        else:
            mask = mask.to(
                dtype=torch.bool,
                device=x.device,
            )

            weights = mask.to(
                dtype=x.dtype,
            ).unsqueeze(-1)

            hidden = (
                (x * weights).sum(dim=1)
                / weights.sum(dim=1).clamp_min(1.0)
            )

        gate = torch.sigmoid(
            self.gate(hidden)
        )

        fast_prediction = self.fast(hidden)
        slow_prediction = self.slow(hidden)

        prediction = (
            (1.0 - gate) * fast_prediction
            + gate * slow_prediction
        )

        return prediction


class BCHRPP(nn.Module):
    """
    Current executable BCHR++ skeleton for MOSI regression.

    Input dimensions:
        text:   768
        audio:    5
        vision:  20

    Output:
        one sentiment regression value for each sample
    """

    def __init__(
        self,
        d=256,
        output_dim=1,
        queries=4,
        text_dim=768,
        audio_dim=5,
        vision_dim=20,
    ):
        super().__init__()

        self.hidden_dim = d
        self.queries = queries

        # Map the three different MOSI feature spaces into one
        # common hidden space.
        self.text_projection = nn.Linear(
            text_dim,
            d,
        )
        self.audio_projection = nn.Linear(
            audio_dim,
            d,
        )
        self.vision_projection = nn.Linear(
            vision_dim,
            d,
        )

        self.evidence_encoder = EvidenceEncoder(
            d=d,
            q=queries,
        )

        self.disentangler = CausalDisentangler(d)

        self.counterfactual = (
            CounterfactualContribution()
        )
        self.competition = CognitiveCompetition()

        self.reasoner = (
            CausalHypergraphTransformer(d)
        )

        self.head = DualArbitration(
            d=d,
            output_dim=output_dim,
        )

    def encode_modality(
        self,
        values,
        projection,
        mask=None,
    ):
        if values.ndim != 3:
            raise ValueError(
                "modality input must have shape "
                "[batch, sequence_length, feature_dim]"
            )

        projected = projection(values)

        evidence = self.evidence_encoder(
            projected,
            mask,
        )

        causal, appearance, nuisance = (
            self.disentangler(evidence)
        )

        if mask is None:
            evidence_mask = torch.ones(
                causal.shape[:2],
                dtype=torch.bool,
                device=causal.device,
            )
        else:
            mask = mask.to(
                dtype=torch.bool,
                device=causal.device,
            )

            # If at least one original timestep is valid, all evidence
            # queries generated for this modality are considered valid.
            modality_available = mask.any(
                dim=1,
                keepdim=True,
            )

            evidence_mask = (
                modality_available.expand(
                    -1,
                    causal.size(1),
                )
            )

        return (
            causal,
            appearance,
            nuisance,
            evidence_mask,
        )

    def forward(
        self,
        text,
        audio,
        vision,
        masks=None,
    ):
        masks = masks or {}

        (
            text_causal,
            _,
            _,
            text_evidence_mask,
        ) = self.encode_modality(
            text,
            self.text_projection,
            masks.get("text"),
        )

        (
            audio_causal,
            _,
            _,
            audio_evidence_mask,
        ) = self.encode_modality(
            audio,
            self.audio_projection,
            masks.get("audio"),
        )

        (
            vision_causal,
            _,
            _,
            vision_evidence_mask,
        ) = self.encode_modality(
            vision,
            self.vision_projection,
            masks.get("vision"),
        )

        evidence = torch.cat(
            [
                text_causal,
                audio_causal,
                vision_causal,
            ],
            dim=1,
        )

        evidence_mask = torch.cat(
            [
                text_evidence_mask,
                audio_evidence_mask,
                vision_evidence_mask,
            ],
            dim=1,
        )

        # At least one modality must be available for each sample.
        if not evidence_mask.any(dim=1).all():
            raise ValueError(
                "a sample cannot have all modalities missing"
            )

        reasoned = self.reasoner(
            evidence,
            mask=evidence_mask,
        )

        prediction = self.head(
            reasoned,
            mask=evidence_mask,
        )

        return prediction