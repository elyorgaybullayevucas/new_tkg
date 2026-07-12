"""
TREA-TKG Model — full architecture in one file.

Single novel mechanism (deliberately not a kitchen sink of unablated tricks):
  AdaptiveTemporalAttention — multi-head attention over an entity's history
  where EACH RELATION has its own learnable exponential time-decay
  (log_decay: Embedding(num_relations, num_heads)), instead of the fixed
  global decay used by DaeMon/CyGNet-style copy mechanisms. Fused with a
  frequency/recency copy score via a learned per-query AdaptiveGate.

Scoring: DistMult-style — score(s,r,o) = (query_repr ⊙ h_r) · h_o, reusing
the existing Xavier-initialized rel_emb table. This replaces an earlier
per-relation bilinear head (W_r ∈ R^{d×d} per relation, 65k params/relation
with only one relation sanely initialized) that was a real source of
early-training instability. `use_inverse=True` gives forward/inverse
relation embedding rows, which is the standard trick that compensates for
DistMult's inability to represent antisymmetric relations on its own.

Reference:
  "TREA-TKG: Temporal Recurrence-Enhanced Attention for Temporal
   Knowledge Graph Forecasting" (unpublished, 2025)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from trea.config import TREAConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Temporal Position Encoding (sinusoidal over Δt)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEncoding(nn.Module):
    """
    Sinusoidal encoding of Δt = query_time − event_time.
    Learnable projection on top so the model can shape it.
    """
    def __init__(self, d_model: int, max_period: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        delta_t: (*, ) long tensor of time differences
        Returns : (*, d_model) float tensor
        """
        d = self.d_model
        device = delta_t.device
        dt = delta_t.float().unsqueeze(-1)          # (*, 1)
        div = torch.exp(
            torch.arange(0, d, 2, device=device).float()
            * -(math.log(self.max_period) / d)
        )                                            # (d//2,)
        enc = torch.zeros(*delta_t.shape, d, device=device)
        enc[..., 0::2] = torch.sin(dt * div)
        enc[..., 1::2] = torch.cos(dt * div[: d // 2])
        return self.proj(enc)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Adaptive Temporal Attention
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveTemporalAttention(nn.Module):
    """
    Multi-head attention over the history sequence with a
    per-relation LEARNABLE temporal decay.

    Key innovation: instead of a fixed exp(-λΔt), each relation r
    has its own decay parameter α_r ≥ 0:
        logit_i  = q · k_i / √d − softplus(α_r) · Δt_i
        weight_i = softmax(logit_i)[masked]

    This lets the model learn that e.g. military events decay fast
    while ontological relations (YAGO) are stable.
    """

    def __init__(self, d: int, num_heads: int, num_relations: int,
                 dropout: float = 0.1):
        super().__init__()
        assert d % num_heads == 0
        self.d = d
        self.nh = num_heads
        self.dh = d // num_heads

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)

        # per-relation decay parameter (log-space for stability)
        self.log_decay = nn.Embedding(num_relations, num_heads)
        nn.init.constant_(self.log_decay.weight, 0.0)   # start at decay=1

        self.time_enc = TemporalEncoding(d)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        query: torch.Tensor,       # (B, d)  — (E[s] + R[r])
        hist_feats: torch.Tensor,  # (B, L, d) — encoded history events
        hist_rels: torch.Tensor,   # (B, L) — relation ids for each hist event
        delta_ts: torch.Tensor,    # (B, L) — Δt for each hist event
        mask: torch.BoolTensor,    # (B, L) — True = valid
    ) -> torch.Tensor:             # (B, d)
        B, L, d = hist_feats.shape

        # ── add temporal encoding to values ──────────────────────────────────
        t_enc = self.time_enc(delta_ts)                  # (B, L, d)
        hist_feats = hist_feats + t_enc

        q = self.W_q(query).view(B, 1, self.nh, self.dh).transpose(1, 2)  # (B,nh,1,dh)
        k = self.W_k(hist_feats).view(B, L, self.nh, self.dh).transpose(1, 2)
        v = self.W_v(hist_feats).view(B, L, self.nh, self.dh).transpose(1, 2)

        # ── dot-product attention scores ─────────────────────────────────────
        scale = math.sqrt(self.dh)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, nh, 1, L)

        # ── per-relation temporal decay ───────────────────────────────────────
        # decay shape: (B, L, nh) → (B, nh, 1, L)
        decay_raw = self.log_decay(hist_rels)                # (B, L, nh)
        decay = F.softplus(decay_raw).permute(0, 2, 1).unsqueeze(2)  # (B,nh,1,L)
        dt_expand = delta_ts.float().unsqueeze(1).unsqueeze(2)        # (B,1,1,L)
        attn = attn - decay * dt_expand

        # ── masking ───────────────────────────────────────────────────────────
        if mask is not None:
            pad = ~mask                                        # (B, L)
            attn = attn.masked_fill(
                pad.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # ── if ALL positions are padding, fall back to uniform ────────────────
        all_pad = (~mask).all(dim=-1, keepdim=True)          # (B, 1)
        attn_weights = torch.softmax(attn, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # ── aggregate values ─────────────────────────────────────────────────
        ctx = torch.matmul(self.drop(attn_weights), v)       # (B, nh, 1, dh)
        ctx = ctx.squeeze(2).transpose(1, 2).reshape(B, d)   # (B, d)
        out = self.W_o(ctx)                                   # (B, d)

        # ── residual + norm ───────────────────────────────────────────────────
        # When all padding, just pass the query through
        out = torch.where(all_pad.expand_as(out), query, out)
        return self.norm(out + query)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Adaptive Gate (structural ↔ copy fusion)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveGate(nn.Module):
    """
    Learned scalar gate  g ∈ (0,1)  per query.
        final_score = g · embed_score + (1−g) · copy_score

    Gate conditioned on query representation (s, r, context).
    """
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, query_repr: torch.Tensor) -> torch.Tensor:
        """query_repr: (B, d)  →  gate: (B, 1)"""
        return torch.sigmoid(self.mlp(query_repr))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main TREA-TKG Model
# ─────────────────────────────────────────────────────────────────────────────

class TREAModel(nn.Module):
    """
    Full TREA-TKG model.

    Forward pass returns logits over ALL entities for the given queries.
    (1-vs-N softmax training)
    """

    def __init__(self, num_entities: int, num_relations: int,
                 cfg: TREAConfig):
        super().__init__()
        d = cfg.embed_dim
        # Relation embedding tables are always sized for forward + inverse
        # (R = num_relations*2), regardless of cfg.use_inverse: GraphIndex
        # (trea/data.py) always builds BOTH views into history/copy lookups
        # (an entity's history includes facts where it was the object, seen
        # as the subject of relation `rel + num_relations`) so hist_rels can
        # contain doubled ids even when training triples themselves aren't
        # inverse-augmented. Sizing for num_relations only would risk an
        # out-of-bounds embedding lookup.
        R = num_relations * 2

        # ── base embeddings ───────────────────────────────────────────────────
        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(R, d)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # ── structural path ───────────────────────────────────────────────────
        self.hist_rel_emb = nn.Embedding(R, d)   # separate embeddings for history
        self.attn = AdaptiveTemporalAttention(d, cfg.num_heads, R, cfg.dropout)

        # Feed-forward after attention
        self.ff = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, d),
        )

        # ── copy path ─────────────────────────────────────────────────────────
        # Learnable temperature per relation for copy score sharpening
        self.copy_temp = nn.Embedding(R, 1)
        nn.init.constant_(self.copy_temp.weight, 1.0)

        # ── fusion ────────────────────────────────────────────────────────────
        self.gate = AdaptiveGate(d, cfg.gate_hidden)

        # ── output scoring ────────────────────────────────────────────────────
        # DistMult: score(s,r,o) = (query_repr ⊙ rel_emb[r]) · ent_emb[o]
        # (rel_emb already used inside encode_query as an additive query cue;
        # here it's reused multiplicatively as the relation-specific scoring
        # transform — no separate weight table needed.)

        self.drop = nn.Dropout(cfg.dropout)
        self.num_entities = num_entities
        self.num_relations_total = R
        self.cfg = cfg

    # ── forward ───────────────────────────────────────────────────────────────

    def encode_query(
        self,
        subs: torch.Tensor,          # (B,)
        rels: torch.Tensor,          # (B,)
        hist_rels: torch.Tensor,     # (B, L)
        hist_objs: torch.Tensor,     # (B, L)
        hist_times: torch.Tensor,    # (B, L)   raw timestamps
        query_times: torch.Tensor,   # (B,)     raw timestamps
        hist_mask: torch.BoolTensor, # (B, L)
    ) -> torch.Tensor:               # (B, d)
        """
        Encode the query into a d-dim representation using
        entity/relation embeddings + adaptive temporal attention.
        """
        d = self.cfg.embed_dim
        B, L = hist_rels.shape

        # base query vector
        h_s = self.ent_emb(subs)        # (B, d)
        h_r = self.rel_emb(rels)        # (B, d)
        query = h_s + h_r               # (B, d)

        if hist_mask.any():
            # encode each historical event as  h_obj + h_hist_rel
            h_hist_obj = self.ent_emb(hist_objs)          # (B, L, d)
            h_hist_rel = self.hist_rel_emb(hist_rels)     # (B, L, d)
            hist_feats = h_hist_obj + h_hist_rel          # (B, L, d)

            # Δt (clamped to ≥ 0)
            delta_t = (query_times.unsqueeze(1) - hist_times).clamp(min=0)  # (B, L)

            ctx = self.attn(query, hist_feats, hist_rels, delta_t, hist_mask)  # (B, d)
        else:
            ctx = query

        # combine query + context via feed-forward
        combined = self.ff(torch.cat([query, ctx], dim=-1))  # (B, d)
        return self.drop(combined)

    def structural_score(
        self,
        query_repr: torch.Tensor,  # (B, d)
        rels: torch.Tensor,        # (B,)
    ) -> torch.Tensor:             # (B, num_entities)
        """
        DistMult scoring:  score[o] = (query_repr ⊙ rel_emb[r]) · h_o
        Stable, standard, O(d) params per relation (vs. the earlier
        per-relation d×d bilinear head).
        """
        r_e = self.rel_emb(rels)                     # (B, d)
        q_transformed = query_repr * r_e              # (B, d) Hadamard product

        all_ents = self.ent_emb.weight               # (N, d)
        scores = torch.matmul(q_transformed, all_ents.t())  # (B, N)
        return scores

    def forward(
        self,
        subs: torch.Tensor,
        rels: torch.Tensor,
        hist_rels: torch.Tensor,
        hist_objs: torch.Tensor,
        hist_times: torch.Tensor,
        query_times: torch.Tensor,
        hist_mask: torch.BoolTensor,
        copy_scores: torch.Tensor,   # (B, N)  pre-computed copy scores
    ) -> torch.Tensor:               # (B, N)  final logits
        """
        Returns final logits (B, num_entities) for all-entity scoring.
        """
        # ── structural path ───────────────────────────────────────────────────
        query_repr = self.encode_query(
            subs, rels, hist_rels, hist_objs,
            hist_times, query_times, hist_mask
        )
        embed_logits = self.structural_score(query_repr, rels)   # (B, N)

        # ── copy path ────────────────────────────────────────────────────────
        # sharpen / scale copy scores with per-relation temperature
        temp = F.softplus(self.copy_temp(rels))                  # (B, 1)
        copy_logits = copy_scores * temp                         # (B, N)

        # ── adaptive fusion ───────────────────────────────────────────────────
        gate = self.gate(query_repr)                             # (B, 1)
        final_logits = gate * embed_logits + (1 - gate) * copy_logits

        return final_logits

    # ── helper: get entity embeddings for contrastive loss ────────────────────
    def get_entity_embeddings(self, entity_ids: torch.Tensor) -> torch.Tensor:
        return self.ent_emb(entity_ids)
