"""
TREA-TKG configuration — all hyperparameters in one place.
"""
import argparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TREAConfig:
    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset: str = "ICEWS18"
    data_dir: str = "data"        # repo-relative data/ dir (see data/<DATASET>/)

    # ── Model ────────────────────────────────────────────────────────────────
    embed_dim: int = 256          # entity / relation embedding size
    history_len: int = 10         # how many past TIMESTEPS to look back
    num_heads: int = 4            # attention heads in temporal attention
    dropout: float = 0.2
    use_inverse: bool = True      # add inverse triples during training

    # ── Adaptive Gate ────────────────────────────────────────────────────────
    gate_hidden: int = 128        # hidden size of gate MLP

    # ── Copy Head ───────────────────────────────────────────────────────────
    copy_lambda: float = 0.5      # time-decay for copy score
    copy_freq_smooth: float = 1.0 # smoothing for log(1 + freq)

    # ── Training ─────────────────────────────────────────────────────────────
    max_epochs: int = 50          # hard safety cap — early stopping usually fires first
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # ── Early stopping ───────────────────────────────────────────────────────
    early_stop_patience:  int   = 8       # eval rounds with no improvement before stopping
    early_stop_min_delta: float = 1e-4    # minimum MRR improvement to reset patience

    # ── Loss ─────────────────────────────────────────────────────────────────
    alpha_contrastive: float = 0.3  # weight of contrastive loss
    margin: float = 2.0             # triplet margin
    label_smoothing: float = 0.1

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_every: int = 1             # evaluate every N epochs
    hits_at: tuple = (1, 3, 10)

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "checkpoints/trea"
    log_dir: str = "logs/trea"
    num_workers: int = 4           # lower this on Windows/CPU dev boxes (spawn, no COW)


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset-specific defaults
# ══════════════════════════════════════════════════════════════════════════════
# ICEWS14 intentionally excluded: data/ICEWS14/ currently contains a
# byte-for-byte duplicate of ICEWS18's files (23033 entities / 256 relations
# instead of ICEWS14's real ~7128/230) — training on it would silently give
# meaningless numbers. Re-add once real ICEWS14 source data is available.
DATASET_CHOICES = ["ICEWS18", "WIKI", "YAGO", "GDELT"]


def apply_dataset_defaults(cfg: TREAConfig, cli: set) -> TREAConfig:
    """
    Per-dataset defaults tuned for each benchmark's size/density, mirroring
    main.py's apply_dataset_config() pattern. CLI-provided values always win
    (tracked via `cli`, the set of arg names that differ from the argparse
    default — see parse_args()).
    """
    def _set(key, val):
        if key not in cli:
            setattr(cfg, key, val)

    if cfg.dataset == "ICEWS18":
        _set("batch_size", 512);  _set("history_len", 16)
        _set("early_stop_patience", 8);  _set("max_epochs", 30)
    elif cfg.dataset in ("WIKI", "YAGO"):
        _set("batch_size", 256);  _set("history_len", 10)
        _set("early_stop_patience", 5);  _set("max_epochs", 60)
    elif cfg.dataset == "GDELT":
        # 1.73M train quads, 15-min granularity — dense graph, keep history
        # window conservative so per-item lookups stay cheap.
        _set("batch_size", 512);  _set("history_len", 16)
        _set("early_stop_patience", 6);  _set("max_epochs", 25)

    return cfg


def parse_args() -> TREAConfig:
    parser = argparse.ArgumentParser(description="TREA-TKG Training")

    parser.add_argument("--dataset", type=str, default="ICEWS18",
                        choices=DATASET_CHOICES)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha_contrastive", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="checkpoints/trea")
    parser.add_argument("--log_dir", type=str, default="logs/trea")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--copy_lambda", type=float, default=0.5)
    parser.add_argument("--no_inverse", dest="use_inverse",
                        action="store_false", default=True)

    args = parser.parse_args()

    # Which args were explicitly given (differ from the parser's own default)
    defaults = {a.dest: a.default for a in parser._actions}
    cli = {k for k, v in vars(args).items()
           if v != defaults.get(k) and k != "dataset"}

    cfg = TREAConfig(**vars(args))
    return apply_dataset_defaults(cfg, cli)
