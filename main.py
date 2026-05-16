# main.py
"""
STORM Model — ishga tushirish.

Ishlatish:
    python main.py --dataset ICEWS18
    python main.py --dataset WIKI
    python main.py --dataset YAGO --epochs 300
    python main.py --resume checkpoints/ICEWS18_best.pt
"""
import argparse, os, random
import numpy as np
import torch

from config import Config, DATASET_STATS
from data.datamodule import TKGDataModule
from models.elite_tkg_model import ORIONModel as EliteTKGModel
from trainers.trainer import EliteTrainer
from utils.logging import get_logger


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args() -> Config:
    p = argparse.ArgumentParser("CATRE — Cross-scale Adaptive Temporal Reasoning for Extrapolation")

    p.add_argument("--dataset",  default="ICEWS18",
                   choices=["ICEWS14", "ICEWS18", "WIKI", "YAGO", "YAGOs", "GDELT"])
    p.add_argument("--data_dir", default="data")

    p.add_argument("--entity_dim",   type=int,   default=256)
    p.add_argument("--relation_dim", type=int,   default=256)
    p.add_argument("--delta_dim",    type=int,   default=64)
    p.add_argument("--hidden_dim",   type=int,   default=512)
    p.add_argument("--num_heads",    type=int,   default=8)
    p.add_argument("--num_layers",   type=int,   default=2)
    p.add_argument("--ffn_dim",      type=int,   default=1024)
    p.add_argument("--dropout",      type=float, default=0.1)

    p.add_argument("--num_paths",    type=int, default=8)
    p.add_argument("--max_path_len", type=int, default=3)
    p.add_argument("--num_negative", type=int, default=256)

    p.add_argument("--batch_size",      type=int,   default=512)
    p.add_argument("--epochs",          type=int,   default=50,  dest="num_epochs")
    p.add_argument("--lr",              type=float, default=3e-4, dest="learning_rate")
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    p.add_argument("--w_link",        type=float, default=1.0)
    p.add_argument("--w_self_adv",    type=float, default=0.5)
    p.add_argument("--w_direct",      type=float, default=0.0)
    p.add_argument("--w_ortho_reg",   type=float, default=0.0)

    p.add_argument("--use_direct_scoring", action="store_true")
    p.add_argument("--use_diachronic",     action="store_true")
    p.add_argument("--use_history",        action="store_true")
    p.add_argument("--max_history",        type=int, default=16)
    p.add_argument("--use_reciprocal",     action="store_true")

    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--log_dir",     default="logs")
    p.add_argument("--resume",      default=None)
    p.add_argument("--no_fp16",     action="store_true")
    p.add_argument("--eval_every",  type=int, default=1)

    args = p.parse_args()
    cfg  = Config()
    # Qaysi argumentlar CLI dan berilganini kuzatamiz
    cli_args = {k for k, v in vars(args).items()
                if v != p.get_default(k) and v is not None}
    for k, v in vars(args).items():
        if k == "no_fp16":
            cfg.fp16 = not v
        elif k in ("use_direct_scoring", "use_diachronic", "use_history", "use_reciprocal") and v:
            setattr(cfg, k, True)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg._cli_args = cli_args   # dataset-override dan himoyalash uchun
    return cfg


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)
    logger = get_logger("main", cfg.log_dir)

    # ── Device sozlash ────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        cfg.device = "cpu"
        cfg.fp16   = False
        logger.warning("CUDA mavjud emas — CPU ishlatiladi (GPU tavsiya etiladi!)")
    else:
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu}  ({mem:.1f} GB)")
        cfg.device = "cuda"

    # ── num_workers: Linux da fork() ishlaydi → ko'proq worker ──────────────
    import os, platform
    if "num_workers" not in getattr(cfg, "_cli_args", set()):
        if platform.system() == "Linux":
            cfg.num_workers = min(8, os.cpu_count() or 4)
        else:
            # Windows da spawn() og'ir → 0
            cfg.num_workers = 0

    logger.info(f"Device: {cfg.device}  |  FP16: {cfg.fp16}  |  num_workers: {cfg.num_workers}")

    # ── Dataset-specific sozlamalar — dm.setup() DAN OLDIN! ──────────────────
    # CLI dan berilgan argumentlar override qilmaydi
    cli = getattr(cfg, "_cli_args", set())
    def _set(key, val):
        if key not in cli:
            setattr(cfg, key, val)

    if cfg.dataset == "GDELT":
        _set("num_paths",          3)
        _set("max_path_len",       2)
        _set("batch_size",         512)
        _set("num_negative",       256)
        cfg.use_history        = True
        _set("max_history",        32)
        cfg.use_direct_scoring = True
        cfg.use_diachronic     = True
        cfg.use_reciprocal     = True
        _set("w_direct",           1.0)
        _set("w_self_adv",         0.5)
        _set("w_ortho_reg",        0.001)
        _set("dropout",            0.1)
        _set("learning_rate",      3e-4)
        _set("num_epochs",         30)
        logger.info(
            "GDELT: num_paths=3, max_path_len=2, batch_size=512, "
            "use_history=True, max_history=32, reciprocal=True, epochs=30"
        )

    elif cfg.dataset in ("WIKI", "YAGO", "YAGOs"):
        _set("num_paths",          8)
        _set("max_path_len",       3)
        _set("batch_size",         256)
        _set("num_negative",       256)
        cfg.use_history        = True
        _set("max_history",        64)
        cfg.use_direct_scoring = True
        cfg.use_diachronic     = True
        cfg.use_reciprocal     = True
        _set("w_direct",           2.0)
        _set("w_self_adv",         0.1)
        _set("w_ortho_reg",        0.001)
        _set("dropout",            0.15)
        _set("learning_rate",      3e-4)
        _set("num_epochs",         500)
        logger.info(
            f"{cfg.dataset}: use_history=True, max_history=64, reciprocal=True, "
            f"DirectScoring=True, Diachronic=True, w_direct=2.0, "
            f"w_self_adv=0.1, LR=3e-4, epochs=500"
        )

    elif cfg.dataset == "ICEWS18":
        _set("num_paths",          8)
        _set("max_path_len",       3)
        _set("batch_size",         512)
        _set("num_negative",       256)
        cfg.use_history        = True
        _set("max_history",        64)
        cfg.use_direct_scoring = True
        cfg.use_diachronic     = True
        cfg.use_reciprocal     = True
        _set("w_direct",           1.0)
        _set("w_self_adv",         0.5)
        _set("w_ortho_reg",        0.001)
        _set("dropout",            0.1)
        _set("learning_rate",      3e-4)
        _set("num_epochs",         50)
        logger.info(
            "ICEWS18: use_history=True, max_history=64, reciprocal=True, "
            "DirectScoring=True, Diachronic=True, w_direct=1.0, epochs=50"
        )

    elif cfg.dataset == "ICEWS14":
        _set("num_paths",          8)
        _set("max_path_len",       3)
        _set("batch_size",         512)
        _set("num_negative",       256)
        cfg.use_history        = True
        _set("max_history",        64)
        cfg.use_direct_scoring = True
        cfg.use_diachronic     = True
        cfg.use_reciprocal     = True
        _set("w_direct",           1.0)
        _set("w_self_adv",         0.5)
        _set("w_ortho_reg",        0.001)
        _set("dropout",            0.1)
        _set("learning_rate",      3e-4)
        _set("num_epochs",         50)
        logger.info(
            "ICEWS14: use_history=True, max_history=64, reciprocal=True, "
            "DirectScoring=True, Diachronic=True, w_direct=1.0, epochs=50"
        )

    # ── DataModule ────────────────────────────────────────────────────────────
    logger.info(f"Dataset yuklanmoqda: {cfg.dataset}  (reciprocal={cfg.use_reciprocal})")
    dm = TKGDataModule(cfg)
    dm.setup()
    logger.info(
        f"Dataset={cfg.dataset} | "
        f"|E|={cfg.num_entities} | "
        f"|R|={cfg.num_relations} | "
        f"|T|={cfg.num_times}"
    )
    logger.info(
        f"Train:{len(dm.train_ds):,}  "
        f"Valid:{len(dm.valid_ds):,}  "
        f"Test:{len(dm.test_ds):,}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EliteTKGModel(
        num_entities        = cfg.num_entities,
        num_relations       = cfg.num_relations,
        num_times           = cfg.num_times,
        entity_dim          = cfg.entity_dim,
        relation_dim        = cfg.relation_dim,
        delta_dim           = cfg.delta_dim,
        hidden_dim          = cfg.hidden_dim,
        num_heads           = cfg.num_heads,
        num_layers          = cfg.num_layers,
        ffn_dim             = cfg.ffn_dim,
        num_negative        = cfg.num_negative,
        num_patterns        = cfg.num_patterns,
        dropout             = cfg.dropout,
        label_smoothing     = cfg.label_smoothing,
        w_direct            = cfg.w_direct,
        w_pattern_div       = cfg.w_pattern_div,
        use_direct_scoring  = cfg.use_direct_scoring,
        use_diachronic      = cfg.use_diachronic,
        use_history         = cfg.use_history,
        max_history         = cfg.max_history,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parametrlar: {n_params/1e6:.2f}M")

    # ── Multi-GPU (DataParallel) ───────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and cfg.device == "cuda":
        model = torch.nn.DataParallel(model)
        logger.info(f"DataParallel: {n_gpus} ta GPU")
    else:
        logger.info(f"Single GPU/CPU")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = EliteTrainer(
        model         = model,
        cfg           = cfg,
        train_loader  = dm.train_loader(),
        valid_loader  = dm.valid_loader(),
        test_loader   = dm.test_loader(),
        valid_dataset = dm.valid_ds,
        test_dataset  = dm.test_ds,
    )

    test_metrics = trainer.fit()
    logger.info("O'qitish yakunlandi!")
    logger.info(f"Test: {test_metrics}")


if __name__ == "__main__":
    main()
