"""
TREA-TKG — main entry point.

Datasets: ICEWS18, WIKI, YAGO, GDELT (ICEWS14 excluded — data/ICEWS14/ is
currently a duplicate of ICEWS18's files, see trea/config.py). Per-dataset
batch size / history length / early-stopping patience / max_epochs defaults
are applied automatically (trea/config.py:apply_dataset_defaults) — training
stops as soon as validation MRR stops improving, not at a fixed epoch count.

Usage examples:
    # Single dataset — per-dataset defaults applied automatically
    python train_trea.py --dataset ICEWS18
    python train_trea.py --dataset WIKI
    python train_trea.py --dataset YAGO
    python train_trea.py --dataset GDELT

    # Override any default explicitly
    python train_trea.py --dataset YAGO --embed_dim 128 --lr 2e-3

    # CPU (no GPU)
    python train_trea.py --dataset YAGO --device cpu --batch_size 64 --num_workers 0
"""
import sys
import os

# Make sure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(__file__))

from trea.config import parse_args
from trea.trainer import TREATrainer


def main():
    cfg = parse_args()

    print(f"""
╔══════════════════════════════════════════════════╗
║          TREA-TKG  (v2 — 2025)                   ║
║  Temporal Recurrence-Enhanced Attention for TKG  ║
╠══════════════════════════════════════════════════╣
║  Dataset   : {cfg.dataset:<35}║
║  embed_dim : {cfg.embed_dim:<35}║
║  history   : {cfg.history_len:<35}║
║  max_epochs: {cfg.max_epochs:<35}║
║  patience  : {cfg.early_stop_patience:<35}║
║  lr        : {cfg.lr:<35}║
║  α_contrast: {cfg.alpha_contrastive:<35}║
║  device    : {cfg.device:<35}║
╚══════════════════════════════════════════════════╝
""")

    trainer = TREATrainer(cfg)
    results = trainer.train()
    return results


if __name__ == "__main__":
    main()
