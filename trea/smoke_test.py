"""
CPU-only correctness smoke test for the TREA-TKG v2 rewrite.

No GPU required. Run from the repo root:

    python -m trea.smoke_test

Checks:
  1. GraphIndex.get_copy_counts() (the rewritten, indexed version) matches a
     brute-force reference on a handcrafted tiny index — regression guard for
     the O(all triples)-per-query bug fix.
  2. End-to-end forward/backward on a tiny synthetic dataset: no shape
     errors, no NaNs, loss decreases over a few epochs (the toy pattern is
     deterministic/trivially memorizable).
  3. evaluate() returns MRR in [0,1] with Hits@10 >= Hits@3 >= Hits@1.

This does NOT validate model quality on real data — that only happens on
the GPU box with the real ICEWS18/WIKI/YAGO/GDELT datasets.
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trea.config import TREAConfig
from trea.data import GraphIndex, TKGDataLoader
from trea.trainer import TREATrainer


def _fail(msg: str):
    print(f"  ✗ FAIL: {msg}")
    sys.exit(1)


def _ok(msg: str):
    print(f"  ✓ {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. get_copy_counts vs. brute-force reference
# ─────────────────────────────────────────────────────────────────────────────

def brute_force_copy_counts(quads_all, num_relations, sub, rel, query_time,
                             num_entities, copy_lambda, step):
    scores = np.zeros(num_entities, dtype=np.float32)
    # Reproduce the ORIGINAL (pre-rewrite) semantics exactly, including the
    # forward+inverse doubling now built into GraphIndex.
    counts = {}
    last_seen = {}
    for s, r, o, t in quads_all:
        s, r, o, t = int(s), int(r), int(o), int(t)
        if t >= query_time:
            continue
        if s == sub and r == rel:
            counts[o] = counts.get(o, 0) + 1
            last_seen[o] = max(last_seen.get(o, -1), t)
        # inverse view: o acting as subject of rel+num_relations
        if o == sub and (r + num_relations) == rel:
            counts[s] = counts.get(s, 0) + 1
            last_seen[s] = max(last_seen.get(s, -1), t)
    for o, c in counts.items():
        recency = np.exp(-copy_lambda * (query_time - last_seen[o]) / max(step, 1))
        scores[o] = np.log1p(c) * recency
    return scores


def test_copy_counts():
    print("\n[1/3] get_copy_counts vs. brute-force reference")
    rng = np.random.default_rng(0)
    num_entities, num_relations = 12, 3
    quads_all = []
    for _ in range(200):
        s = rng.integers(0, num_entities)
        r = rng.integers(0, num_relations)
        o = rng.integers(0, num_entities)
        t = rng.integers(0, 50)
        quads_all.append([s, r, o, t])
    quads_all = np.array(quads_all, dtype=np.int32)

    index = GraphIndex(quads_all, num_relations, step=1)

    for trial in range(20):
        sub = int(rng.integers(0, num_entities))
        rel = int(rng.integers(0, num_relations * 2))   # include inverse ids
        qt  = int(rng.integers(1, 50))
        got = index.get_copy_counts(sub, rel, qt, num_entities, copy_lambda=0.5, step=1)
        ref = brute_force_copy_counts(quads_all, num_relations, sub, rel, qt,
                                      num_entities, copy_lambda=0.5, step=1)
        if not np.allclose(got, ref, atol=1e-5):
            _fail(f"mismatch at trial {trial} (sub={sub}, rel={rel}, t={qt}): "
                  f"got={got}, ref={ref}")
    _ok("indexed get_copy_counts matches brute-force reference on 20 random queries")


# ─────────────────────────────────────────────────────────────────────────────
# 2 & 3. End-to-end tiny synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────

def _write_toy_dataset(base_dir: str):
    """20 entities, 4 relations, 30 timesteps, deterministic pattern:
    (e, e%4, (e + e%4 + 1) % 20, t) for every t — trivially memorizable,
    time-invariant so any train/valid/test time split is learnable."""
    num_entities, num_relations, num_times = 20, 4, 30
    lines = {"train": [], "valid": [], "test": []}
    for t in range(num_times):
        split = "train" if t < 20 else ("valid" if t < 25 else "test")
        for e in range(num_entities):
            r = e % num_relations
            o = (e + r + 1) % num_entities
            lines[split].append(f"{e}\t{r}\t{o}\t{t}")

    os.makedirs(os.path.join(base_dir, "TOY"), exist_ok=True)
    for split, rows in lines.items():
        with open(os.path.join(base_dir, "TOY", f"{split}.txt"), "w") as f:
            f.write("\n".join(rows) + "\n")


def test_end_to_end():
    print("\n[2/3]+[3/3] end-to-end forward/backward + evaluate() on toy data")
    tmp_dir = tempfile.mkdtemp(prefix="trea_smoke_")
    try:
        _write_toy_dataset(tmp_dir)

        cfg = TREAConfig(
            dataset="TOY", data_dir=tmp_dir,
            embed_dim=16, num_heads=2, gate_hidden=8,
            history_len=4, batch_size=8, num_workers=0,
            lr=5e-3, max_epochs=15, eval_every=3,
            early_stop_patience=100,   # don't stop early in the smoke test
            device="cpu",
            save_dir=os.path.join(tmp_dir, "ckpt"),
            log_dir=os.path.join(tmp_dir, "logs"),
        )

        trainer = TREATrainer(cfg)

        # ── run a couple of raw epochs manually to inspect loss + NaNs ──────
        losses = []
        for epoch in range(1, 6):
            loss_d = trainer._train_epoch(epoch)
            if not np.isfinite(loss_d["total"]):
                _fail(f"non-finite loss at epoch {epoch}: {loss_d}")
            losses.append(loss_d["total"])
        _ok(f"5 epochs ran with finite losses: {[f'{l:.4f}' for l in losses]}")

        if not (losses[-1] < losses[0]):
            _fail(f"loss did not decrease on a trivially memorizable toy set: "
                  f"first={losses[0]:.4f} last={losses[-1]:.4f}")
        _ok(f"loss decreased ({losses[0]:.4f} -> {losses[-1]:.4f})")

        # ── full train() loop (exercises early stopping + checkpoint + eval) ─
        test_metrics = trainer.train()

        mrr = test_metrics["MRR"]
        if not (0.0 <= mrr <= 1.0):
            _fail(f"MRR out of bounds: {mrr}")
        h1, h3, h10 = test_metrics["Hits@1"], test_metrics["Hits@3"], test_metrics["Hits@10"]
        if not (h1 <= h3 <= h10):
            _fail(f"Hits@k not monotonic: H@1={h1} H@3={h3} H@10={h10}")
        _ok(f"evaluate() sane: MRR={mrr:.4f} H@1={h1:.4f} H@3={h3:.4f} H@10={h10:.4f}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Windows consoles default to cp1252, which can't encode the ✓/✗ glyphs
    # used below; force UTF-8 stdout (real training runs on Linux, UTF-8 by
    # default, so this only matters for local smoke testing).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("  TREA-TKG v2 — CPU smoke test")
    print("=" * 70)
    test_copy_counts()
    test_end_to_end()
    print("\n" + "=" * 70)
    print("  ALL SMOKE TESTS PASSED")
    print("=" * 70)
