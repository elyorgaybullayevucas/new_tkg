"""
Data loading, graph indexing, and PyTorch Dataset for TREA-TKG.

Quadruple format (tab/space-separated):
    sub_id  rel_id  obj_id  timestamp  [0]

Supports ICEWS18 (step=24), WIKI (step=1), YAGO (step=1), GDELT (step=15).
"""
import os
import sys
import bisect
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from trea.config import TREAConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_quadruples(path: str) -> np.ndarray:
    """Return (N, 4) int32 array  [sub, rel, obj, time]."""
    quads = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            quads.append([int(parts[0]), int(parts[1]),
                          int(parts[2]), int(parts[3])])
    return np.array(quads, dtype=np.int32)


def load_id_map(path: str) -> Dict[str, int]:
    """entity2id.txt or relation2id.txt  →  {name: id}."""
    id_map: Dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                id_map[parts[0]] = int(parts[1])
    return id_map


def get_dataset_step(dataset: str) -> int:
    """Timestamp increment per timestep for each dataset (verified from raw files)."""
    if dataset in ("ICEWS18", "ICEWS14"):
        return 24
    if dataset == "GDELT":
        return 15
    return 1   # WIKI, YAGO — already unit-step timestamps


# Known-good stats to catch silently-duplicated/corrupted dataset dirs
# (e.g. data/ICEWS14 was found to be a byte-for-byte copy of ICEWS18).
_EXPECTED_STATS = {
    "ICEWS18": (23033, 256),
    "WIKI":    (12554, 24),
    "YAGO":    (10623, 10),
    "GDELT":   (7691, 240),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Graph Index — fast history look-ups
# ─────────────────────────────────────────────────────────────────────────────

class GraphIndex:
    """
    Pre-built indices over all known quadruples for:
      a) filtered evaluation (all_answers[(s, r, t)] → set of true objects)
      b) history retrieval   (facts_before[t][s]    → list of (rel, obj, time)),
         including the inverse view (entity as object → treated as subject of
         relation `rel + num_relations`), mirroring utils/paths.py's build_graph
         convention so inverse-augmented queries get real history instead of none.
      c) copy score          (sr_index[(s, r)][o]    → sorted np.ndarray of times)
         also built for both forward and inverse relations.

    Built once from train + valid + test at construction time.
    O(n) construction; O(log deg) / O(deg(s,r)) per-query lookups — no
    per-query scan over the whole dataset (that was the bug in the previous
    version: get_copy_counts() used to iterate every unique (s,r,o) triple in
    the dataset for every single query).
    """

    def __init__(self, quads_all: np.ndarray, num_relations: int, step: int = 1):
        self.step = step
        self.num_relations = num_relations
        self.unique_times = np.unique(quads_all[:, 3])

        # filtered answers: (s, r, t) → {o1, o2, ...}  (raw relations only —
        # used for eval filtering, never sees inverse-augmented relation ids)
        self.all_answers: Dict[Tuple[int, int, int], set] = defaultdict(set)
        for s, r, o, t in quads_all:
            self.all_answers[(int(s), int(r), int(t))].add(int(o))

        # facts indexed by (time, subject) → list[(rel, obj, time)]
        # includes BOTH the forward view (s was subject) and the inverse view
        # (o was object → treated as subject of rel+num_relations)
        by_time_sub: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = \
            defaultdict(list)

        # copy counter per (s, r) → {o: [t1, t2, ...]}, also forward + inverse
        sr_tmp: Dict[Tuple[int, int], Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))

        for s, r, o, t in quads_all:
            s, r, o, t = int(s), int(r), int(o), int(t)
            by_time_sub[(t, s)].append((r, o, t))
            by_time_sub[(t, o)].append((r + num_relations, s, t))
            sr_tmp[(s, r)][o].append(t)
            sr_tmp[(o, r + num_relations)][s].append(t)

        self._by_time_sub = by_time_sub
        # Plain sorted Python lists, not np.array: per-query lookups use
        # bisect.bisect_left on a handful of elements (a (s,r) pair rarely
        # repeats to more than a few dozen objects) — profiling showed
        # np.searchsorted's per-call dispatch overhead dominates at this
        # scale (thousands of tiny numpy calls/batch); plain bisect is
        # substantially cheaper here.
        self._sr_index: Dict[Tuple[int, int], Dict[int, List[int]]] = {
            key: {o: sorted(ts) for o, ts in objs.items()}
            for key, objs in sr_tmp.items()
        }

    # ── history window ────────────────────────────────────────────────────────

    def get_history(
        self,
        sub: int,
        query_time: int,
        history_len: int,
    ) -> List[Tuple[int, int, int]]:
        """
        Return up to history_len × step facts for *subject* sub
        in timesteps [query_time - history_len*step, query_time).
        Returns list of (rel, obj, time) sorted newest-first.
        `rel` may be >= num_relations (inverse view) — see class docstring.
        """
        facts: List[Tuple[int, int, int]] = []
        t = query_time - self.step
        steps_back = 0
        while steps_back < history_len and t >= 0:
            facts.extend(self._by_time_sub.get((t, sub), []))
            t -= self.step
            steps_back += 1
        # Sort newest-first and cap
        facts.sort(key=lambda x: -x[2])
        return facts[:history_len * 10]   # keep max 10×history_len facts

    # ── copy score vector ─────────────────────────────────────────────────────

    def get_copy_counts(
        self,
        sub: int,
        rel: int,
        query_time: int,
        num_entities: int,
        copy_lambda: float,
        step: int,
    ) -> np.ndarray:
        """
        Build a copy-score vector of shape (num_entities,).

        copy[o] = log(1 + count(s,r,o,<t)) × exp(−λ × (query_time − last_seen) / step)

        `rel` may be an inverse-augmented id (rel >= num_relations) — the
        index was built with both views, so this works for augmented
        training triples too, not just raw ones.

        O(distinct objects ever seen for this exact (sub, rel) pair) instead
        of O(all unique triples in the dataset).

        Dense convenience wrapper around get_copy_scores_sparse(); prefer the
        sparse version in hot per-item paths (see TKGDataset.__getitem__) —
        allocating a fresh (num_entities,) array per query is itself a real
        cost at batch_size~512 × num_entities~23k.
        """
        scores = np.zeros(num_entities, dtype=np.float32)
        for o, val in self.get_copy_scores_sparse(sub, rel, query_time, copy_lambda, step):
            if o < num_entities:
                scores[o] = val
        return scores

    def get_copy_scores_sparse(
        self,
        sub: int,
        rel: int,
        query_time: int,
        copy_lambda: float,
        step: int,
    ) -> List[Tuple[int, float]]:
        """Same scoring as get_copy_counts(), but returns [(obj, score), ...]
        instead of a dense vector — cheap to build and cheap to pickle across
        DataLoader worker processes."""
        obj_times = self._sr_index.get((sub, rel))
        if not obj_times:
            return []
        denom = max(step, 1)
        out: List[Tuple[int, float]] = []
        for o, times_list in obj_times.items():
            idx = bisect.bisect_left(times_list, query_time)  # count(t < query_time)
            if idx == 0:
                continue
            last_t = times_list[idx - 1]
            recency = math.exp(-copy_lambda * (query_time - last_t) / denom)
            out.append((o, math.log1p(idx) * recency))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TKGDataset(Dataset):
    """
    Each sample is one query (s, r, o, t) plus its precomputed history and
    copy-score vector. All the expensive per-item work happens here, inside
    __getitem__, so DataLoader(num_workers=...) parallelizes it across
    worker processes instead of blocking the main training loop.
    """

    def __init__(
        self,
        quads: np.ndarray,
        index: GraphIndex,
        num_entities: int,
        history_len: int,
        copy_lambda: float,
        step: int,
        use_inverse: bool = True,
        num_relations: Optional[int] = None,
    ):
        self.use_inverse = use_inverse
        self.num_relations = num_relations
        self.index = index
        self.num_entities = num_entities
        self.history_len = history_len
        self.copy_lambda = copy_lambda
        self.step = step

        if use_inverse and num_relations is not None:
            inv = np.stack([quads[:, 2],
                            quads[:, 1] + num_relations,
                            quads[:, 0],
                            quads[:, 3]], axis=1).astype(np.int32)
            self.data = np.concatenate([quads, inv], axis=0)
        else:
            self.data = quads.copy()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        s, r, o, t = (int(x) for x in self.data[idx])

        facts = self.index.get_history(s, t, self.history_len)
        copy_sc = self.index.get_copy_scores_sparse(
            s, r, t, self.copy_lambda, self.step
        )

        return {
            "subject":     s,
            "relation":    r,
            "object":      o,
            "time":        t,
            "history":     facts,     # list[(rel, obj, time)], rel may be doubled
            "copy_scores": copy_sc,   # list[(obj, score)] — sparse, scattered in collate_fn
        }


def collate_fn(batch: List[Dict], max_per_query: int, num_entities: int) -> Dict[str, torch.Tensor]:
    """
    Pads variable-length per-item history into dense (B, max_per_query)
    tensors and scatters sparse copy scores into a dense (B, num_entities)
    tensor. Must be a module-level function (not a closure/lambda) so it is
    picklable for DataLoader multiprocessing (spawn, e.g. on Windows).

    Copy scores are kept sparse ([(obj, score), ...]) all the way up to this
    point — building a fresh (num_entities,) zero array per item inside
    __getitem__ (512 allocations/batch at ICEWS18 scale) was a measurable
    bottleneck; a single (B, num_entities) zero array + one scatter here is
    much cheaper, and the sparse per-item payload is also cheaper to pickle
    across DataLoader worker processes.

    NOTE: this scatter is built with plain numpy arrays, not per-element
    torch tensor assignment (`t[i, j] = x`) — profiling showed torch's
    scalar `__setitem__` is ~30-40x slower than numpy's for this access
    pattern (Python-level dispatch overhead per element dominates at
    batch_size~512 × up to a few hundred history/copy entries per item),
    which was the actual bottleneck (not __getitem__ or searchsorted).
    """
    B = len(batch)
    subs  = torch.tensor([b["subject"]  for b in batch], dtype=torch.long)
    rels  = torch.tensor([b["relation"] for b in batch], dtype=torch.long)
    objs  = torch.tensor([b["object"]   for b in batch], dtype=torch.long)
    times = torch.tensor([b["time"]     for b in batch], dtype=torch.long)

    h_rels_np  = np.zeros((B, max_per_query), dtype=np.int64)
    h_objs_np  = np.zeros((B, max_per_query), dtype=np.int64)
    h_times_np = np.zeros((B, max_per_query), dtype=np.int64)
    h_mask_np  = np.zeros((B, max_per_query), dtype=np.bool_)
    copy_sc_np = np.zeros((B, num_entities), dtype=np.float32)
    for i, item in enumerate(batch):
        facts = item["history"][:max_per_query]
        if facts:
            fr, fo, ft = zip(*facts)
            n = len(facts)
            h_rels_np[i, :n]  = fr
            h_objs_np[i, :n]  = fo
            h_times_np[i, :n] = ft
            h_mask_np[i, :n]  = True
        cs = [(o, v) for o, v in item["copy_scores"] if o < num_entities]
        if cs:
            objs_i, vals_i = zip(*cs)
            copy_sc_np[i, np.array(objs_i, dtype=np.int64)] = vals_i

    h_rels  = torch.from_numpy(h_rels_np)
    h_objs  = torch.from_numpy(h_objs_np)
    h_times = torch.from_numpy(h_times_np)
    h_mask  = torch.from_numpy(h_mask_np)
    copy_sc = torch.from_numpy(copy_sc_np)

    return {
        "subject": subs, "relation": rels, "object": objs, "time": times,
        "hist_rels": h_rels, "hist_objs": h_objs, "hist_times": h_times, "hist_mask": h_mask,
        "copy_scores": copy_sc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  High-level loader used by the Trainer
# ─────────────────────────────────────────────────────────────────────────────

class TKGDataLoader:
    """
    Loads all splits, builds GraphIndex, exposes train/valid/test datasets
    and ready-to-use DataLoaders (with the picklable collate_fn wired in).
    """

    def __init__(self, cfg: TREAConfig):
        base = os.path.join(cfg.data_dir, cfg.dataset)
        self.cfg = cfg
        self.step = get_dataset_step(cfg.dataset)

        train_q = load_quadruples(os.path.join(base, "train.txt"))
        valid_q = load_quadruples(os.path.join(base, "valid.txt"))
        test_q  = load_quadruples(os.path.join(base, "test.txt"))

        # Infer num_entities / num_relations from data
        all_q = np.concatenate([train_q, valid_q, test_q], axis=0)
        self.num_entities  = int(all_q[:, [0, 2]].max()) + 1
        self.num_relations = int(all_q[:, 1].max()) + 1
        self.num_timestamps = int(all_q[:, 3].max() // self.step) + 1

        expected = _EXPECTED_STATS.get(cfg.dataset)
        if expected is not None and (self.num_entities, self.num_relations) != expected:
            print(
                f"[WARN] {cfg.dataset}: expected (entities, relations)="
                f"{expected}, got ({self.num_entities}, {self.num_relations}) — "
                f"data/{cfg.dataset}/ may be corrupted or mismatched "
                f"(this exact symptom is how the ICEWS14==ICEWS18 duplicate "
                f"data bug was found). Double-check the source files.",
                file=sys.stderr,
            )

        # Build the shared index (train + valid + test for filtered eval)
        self.index = GraphIndex(all_q, self.num_relations, step=self.step)

        # Datasets (inverse triples only for training)
        self.train_set = TKGDataset(
            train_q, self.index, self.num_entities, cfg.history_len,
            cfg.copy_lambda, self.step,
            use_inverse=cfg.use_inverse, num_relations=self.num_relations,
        )
        self.valid_set = TKGDataset(
            valid_q, self.index, self.num_entities, cfg.history_len,
            cfg.copy_lambda, self.step, use_inverse=False,
        )
        self.test_set = TKGDataset(
            test_q, self.index, self.num_entities, cfg.history_len,
            cfg.copy_lambda, self.step, use_inverse=False,
        )

        print(
            f"[{cfg.dataset}] entities={self.num_entities:,}  "
            f"relations={self.num_relations}  step={self.step}  "
            f"train={len(self.train_set):,}  "
            f"valid={len(self.valid_set):,}  "
            f"test={len(self.test_set):,}"
        )

    def _make_loader(self, ds: TKGDataset, shuffle: bool, batch_size: int) -> DataLoader:
        max_per_query = self.cfg.history_len * 10
        from functools import partial
        fn = partial(collate_fn, max_per_query=max_per_query, num_entities=self.num_entities)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            persistent_workers=(self.cfg.num_workers > 0),
            pin_memory=(self.cfg.device == "cuda"),
            collate_fn=fn,
            drop_last=shuffle,
        )

    def train_loader(self) -> DataLoader:
        return self._make_loader(self.train_set, shuffle=True, batch_size=self.cfg.batch_size)

    def valid_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        return self._make_loader(self.valid_set, shuffle=False, batch_size=batch_size or self.cfg.batch_size)

    def test_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        return self._make_loader(self.test_set, shuffle=False, batch_size=batch_size or self.cfg.batch_size)
