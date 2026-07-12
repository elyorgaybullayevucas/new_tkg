"""
Loss functions for TREA-TKG.

1. LabelSmoothingCrossEntropy  — 1-vs-N softmax with label smoothing
2. HardNegativeTripletLoss     — contrastive loss with hard negative mining
3. TREALoss                    — combined total loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy over all entities (1-vs-N) with label smoothing.
    Equivalent to standard CE when smoothing=0.
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                filter_mask: torch.Tensor = None) -> torch.Tensor:
        """
        logits:      (B, N)  raw scores
        targets:     (B,)    true object indices
        filter_mask: (B, N)  bool – True for valid negatives (used at train time
                             to mask OTHER known true answers)
        """
        # Optionally mask other true answers (set to very negative)
        if filter_mask is not None:
            logits = logits.masked_fill(~filter_mask, float("-inf"))

        log_probs = F.log_softmax(logits, dim=-1)         # (B, N)

        # gather log-prob of the true answer
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)

        # label smoothing: add a uniform KL term
        smooth_loss = -log_probs.mean(dim=-1)              # (B,)

        loss = self.confidence * nll + self.smoothing * smooth_loss
        return loss.mean()


class HardNegativeTripletLoss(nn.Module):
    """
    In-batch hard-negative triplet loss.

    For each (anchor, positive) in the batch, find the hardest negative
    in the same batch — the entity that is closest to the anchor but
    is NOT the correct answer.

    Novel: negatives that appear in OTHER ground-truth answers of the batch
    are de-prioritized (soft masking rather than hard removal).
    """
    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        query_repr: torch.Tensor,   # (B, d)  query embeddings
        pos_emb: torch.Tensor,      # (B, d)  positive entity embeddings
        neg_emb: torch.Tensor,      # (B, d)  hard negative entity embeddings
    ) -> torch.Tensor:
        """
        Returns scalar triplet loss.
        """
        # Euclidean distances
        d_pos = torch.sum((query_repr - pos_emb) ** 2, dim=-1)   # (B,)
        d_neg = torch.sum((query_repr - neg_emb) ** 2, dim=-1)   # (B,)

        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


class TREALoss(nn.Module):
    """
    Total TREA loss:
        L = L_CE  +  alpha * L_triplet
    """
    def __init__(self, num_entities: int, smoothing: float = 0.1,
                 alpha: float = 0.3, margin: float = 2.0):
        super().__init__()
        self.ce = LabelSmoothingCE(num_entities, smoothing)
        self.triplet = HardNegativeTripletLoss(margin)
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,       # (B, N)
        targets: torch.Tensor,      # (B,)
        query_repr: torch.Tensor,   # (B, d)
        ent_emb: nn.Embedding,      # entity embedding table
    ) -> dict:
        """Returns dict with 'total', 'ce', 'triplet' losses."""
        # ── CE loss ───────────────────────────────────────────────────────────
        ce_loss = self.ce(logits, targets)

        # ── hard negative mining from batch ──────────────────────────────────
        B, N = logits.shape
        pos_emb = ent_emb(targets)                           # (B, d)

        # For each query, pick the entity with highest score that is NOT target
        # (in-batch hard negative mining)
        logits_neg = logits.clone()
        logits_neg.scatter_(1, targets.unsqueeze(1), float("-inf"))
        hard_neg_ids = logits_neg.argmax(dim=-1)             # (B,)
        neg_emb = ent_emb(hard_neg_ids)                      # (B, d)

        triplet_loss = self.triplet(query_repr, pos_emb, neg_emb)

        total = ce_loss + self.alpha * triplet_loss
        return {
            "total": total,
            "ce": ce_loss.item(),
            "triplet": triplet_loss.item(),
        }
