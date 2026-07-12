"""
Filtered evaluation: MRR, Hits@1, Hits@3, Hits@10.

Standard protocol for TKG forecasting:
  For each test quadruple (s, r, o_true, t):
    1. Score all entities.
    2. Remove scores of OTHER known-true answers (filter).
    3. Rank o_true among the remaining N scores.
    4. Accumulate 1/rank → MRR, (rank≤k) → Hits@k.

Vectorized via utils.metrics (shared with the main ORION pipeline) instead
of a per-item Python loop — the previous version did a Python `for i in
range(B)` with a `.item()` GPU→CPU sync per query, which is slow at scale.
"""
import torch
from typing import Dict, List

from trea.model import TREAModel
from trea.data import TKGDataLoader
from utils.metrics import compute_ranks, ranks_to_metrics


def _build_filter_mask(
    index,
    subjects: torch.Tensor,
    relations: torch.Tensor,
    times: torch.Tensor,
    targets: torch.Tensor,
    num_entities: int,
) -> torch.Tensor:
    """(B, num_entities) bool — True for OTHER known-true answers to mask out."""
    B = subjects.size(0)
    mask = torch.zeros(B, num_entities, dtype=torch.bool)
    for i in range(B):
        key = (int(subjects[i]), int(relations[i]), int(times[i]))
        tgt = int(targets[i])
        for obj in index.all_answers.get(key, ()):
            if obj != tgt and 0 <= obj < num_entities:
                mask[i, obj] = True
    return mask


@torch.no_grad()
def evaluate(
    model: TREAModel,
    loader: TKGDataLoader,
    split: str = "valid",            # "valid" | "test"
    device: torch.device = torch.device("cpu"),
    batch_size: int = 256,
    hits_at: tuple = (1, 3, 10),
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute filtered MRR and Hits@k on the requested split."""
    model.eval()

    dl = loader.valid_loader(batch_size) if split == "valid" else loader.test_loader(batch_size)
    num_entities = loader.num_entities
    index = loader.index

    all_ranks: List[torch.Tensor] = []

    from tqdm import tqdm
    desc = f"Evaluating [{split}]"
    for batch in tqdm(dl, desc=desc, disable=not verbose):
        subs    = batch["subject"].to(device)
        rels    = batch["relation"].to(device)
        objs    = batch["object"].to(device)
        times   = batch["time"].to(device)
        h_rels  = batch["hist_rels"].to(device)
        h_objs  = batch["hist_objs"].to(device)
        h_times = batch["hist_times"].to(device)
        h_mask  = batch["hist_mask"].to(device)
        copy_sc = batch["copy_scores"].to(device)

        logits = model(subs, rels, h_rels, h_objs, h_times, times, h_mask, copy_sc)  # (B, N)

        filter_mask = _build_filter_mask(
            index, batch["subject"], batch["relation"], batch["time"], batch["object"],
            num_entities,
        ).to(device)

        ranks = compute_ranks(logits, objs, filter_mask, filter_flag=True)
        all_ranks.append(ranks.float().cpu())

    all_ranks_t = torch.cat(all_ranks)
    return ranks_to_metrics(all_ranks_t, list(hits_at))


def format_metrics(metrics: Dict[str, float], epoch: int = None,
                   split: str = "valid") -> str:
    """Pretty-print metrics for a given epoch."""
    prefix = f"[Epoch {epoch:03d}] " if epoch is not None else ""
    split_str = split.upper()
    parts = [f"{split_str}"]
    for k, v in metrics.items():
        parts.append(f"{k}: {v:.4f}")
    return prefix + "  ".join(parts)
