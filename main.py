"""
ORION — Temporal Knowledge Graph Link Prediction
=================================================
Ishlatish:
    python main.py                     # interaktiv menyu
    python main.py ICEWS18             # to'g'ridan-to'g'ri dataset
    python main.py ICEWS18 --epochs 30 # qo'shimcha parametrlar
    python main.py ICEWS18 --resume checkpoints/ICEWS18_best.pt
"""

import argparse
import os
import platform
import random
import sys

import numpy as np
import torch

from config import Config
from data.datamodule import TKGDataModule
from models.elite_tkg_model import ORIONModel as EliteTKGModel
from trainers.trainer import EliteTrainer
from utils.logging import get_logger

# ── Barcha qo'llab-quvvatlanadigan datasetlar ─────────────────────────────────
KNOWN_DATASETS = ["ICEWS14", "ICEWS18", "WIKI", "YAGO", "YAGOs", "GDELT"]


# ══════════════════════════════════════════════════════════════════════════════
#  Yordamchi funksiyalar
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def detect_datasets(data_dir: str) -> list:
    """data/ ichida mavjud datasetlarni topadi."""
    found = []
    for name in KNOWN_DATASETS:
        path = os.path.join(data_dir, name)
        train = os.path.join(path, "train.txt")
        if os.path.isdir(path) and os.path.isfile(train):
            found.append(name)
    return found


def choose_dataset(data_dir: str, cli_dataset: str | None) -> str:
    """
    Dataset tanlash:
      1. CLI dan berilgan bo'lsa → uni ishlatadi
      2. Faqat 1 ta topilsa → avtomatik tanlaydi
      3. Bir nechtasi topilsa → interaktiv menyu
      4. Hech narsa topilmasa → xato va yo'riqnoma
    """
    available = detect_datasets(data_dir)

    # CLI dan berilgan
    if cli_dataset:
        if cli_dataset not in KNOWN_DATASETS:
            print(f"[XATO] Noma'lum dataset: {cli_dataset}")
            print(f"       Mavjudlari: {KNOWN_DATASETS}")
            sys.exit(1)
        if cli_dataset not in available:
            print(f"[XATO] '{cli_dataset}' topilmadi: {data_dir}/{cli_dataset}/train.txt yo'q")
            sys.exit(1)
        return cli_dataset

    # Hech narsa yo'q
    if not available:
        _print_no_data_error(data_dir)
        sys.exit(1)

    # Faqat bitta
    if len(available) == 1:
        print(f"  >> Dataset avtomatik tanlandi: {available[0]}\n")
        return available[0]

    # Menyu
    return _interactive_menu(available)


def _interactive_menu(datasets: list) -> str:
    """Rangali interaktiv menyu."""
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    w = 52
    print(f"\n{BOLD}{'='*w}{RESET}")
    print(f"{BOLD}  ORION — Dataset tanlang{RESET}")
    print(f"{'='*w}")
    for i, ds in enumerate(datasets, 1):
        print(f"  {CYAN}{i}{RESET}.  {ds}")
    print(f"{'='*w}")

    while True:
        try:
            raw = input(f"  Raqam kiriting [1-{len(datasets)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(datasets):
                chosen = datasets[idx]
                print(f"\n  {GREEN}>> {chosen} tanlandi{RESET}\n")
                return chosen
        except (ValueError, EOFError):
            pass
        print(f"  1 dan {len(datasets)} gacha raqam kiriting.")


def _print_no_data_error(data_dir: str):
    print(f"""
[XATO] '{data_dir}/' papkasida dataset topilmadi.

Quyidagilardan kamida bittasi bo'lishi kerak:
  {data_dir}/ICEWS18/train.txt
  {data_dir}/WIKI/train.txt
  {data_dir}/YAGO/train.txt
  {data_dir}/GDELT/train.txt

Ma'lumotni serverga ko'chirish:
  scp data.zip user@server:/path/to/new_tkg/
  unzip data.zip
""")


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset-specific optimal sozlamalar
# ══════════════════════════════════════════════════════════════════════════════

def apply_dataset_config(cfg: Config, cli: set):
    """
    Har bir dataset uchun ilmiy jihatdan optimal sozlamalar.
    CLI dan berilgan argumentlar ustunlik qiladi (_cli_args orqali).
    """
    def _set(key, val):
        if key not in cli:
            setattr(cfg, key, val)

    # ── Umumiy flaglar (CLI override yo'q) ────────────────────────────────────
    cfg.use_history        = True
    cfg.use_direct_scoring = True
    cfg.use_diachronic     = True
    cfg.use_reciprocal     = True

    if cfg.dataset == "ICEWS18":
        _set("num_paths",        8);    _set("max_path_len",     3)
        _set("batch_size",       512);  _set("num_negative",     256)
        _set("max_history",      64);   _set("w_direct",         1.0)
        _set("w_self_adv",       0.5);  _set("w_ortho_reg",      0.001)
        _set("dropout",          0.1);  _set("learning_rate",    3e-4)
        _set("w_copy",           0.5);  _set("w_hist_contrast",  0.3)
        _set("num_epochs",       50)

    elif cfg.dataset == "ICEWS14":
        _set("num_paths",        8);    _set("max_path_len",     3)
        _set("batch_size",       512);  _set("num_negative",     256)
        _set("max_history",      64);   _set("w_direct",         1.0)
        _set("w_self_adv",       0.5);  _set("w_ortho_reg",      0.001)
        _set("dropout",          0.1);  _set("learning_rate",    3e-4)
        _set("w_copy",           0.5);  _set("w_hist_contrast",  0.3)
        _set("num_epochs",       50)

    elif cfg.dataset in ("WIKI", "YAGO", "YAGOs"):
        _set("num_paths",        8);    _set("max_path_len",     3)
        _set("batch_size",       256);  _set("num_negative",     256)
        _set("max_history",      64);   _set("w_direct",         2.0)
        _set("w_self_adv",       0.1);  _set("w_ortho_reg",      0.001)
        _set("dropout",          0.15); _set("learning_rate",    3e-4)
        _set("w_copy",           2.0);  _set("w_hist_contrast",  0.5)  # copy signal critical for WIKI/YAGO
        _set("num_epochs",       500)

    elif cfg.dataset == "GDELT":
        _set("num_paths",        3);    _set("max_path_len",     2)
        _set("batch_size",       512);  _set("num_negative",     256)
        _set("max_history",      32);   _set("w_direct",         1.0)
        _set("w_self_adv",       0.5);  _set("w_ortho_reg",      0.001)
        _set("dropout",          0.1);  _set("learning_rate",    3e-4)
        _set("w_copy",           1.0);  _set("w_hist_contrast",  0.3)
        _set("num_epochs",       30)


# ══════════════════════════════════════════════════════════════════════════════
#  Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="ORION — Temporal Knowledge Graph Link Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset — positional (ixtiyoriy). Yo'q bo'lsa menyu chiqadi.
    p.add_argument("dataset", nargs="?", default=None,
                   choices=KNOWN_DATASETS + [None],
                   help="Dataset nomi (yo'q bo'lsa interaktiv menyu)")
    p.add_argument("--data_dir", default="data",
                   help="Dataset papkasi")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Model parametrlari")
    g.add_argument("--entity_dim",   type=int,   default=256)
    g.add_argument("--relation_dim", type=int,   default=256)
    g.add_argument("--delta_dim",    type=int,   default=64)
    g.add_argument("--hidden_dim",   type=int,   default=512)
    g.add_argument("--num_heads",    type=int,   default=8)
    g.add_argument("--num_layers",   type=int,   default=2)
    g.add_argument("--ffn_dim",      type=int,   default=1024)
    g.add_argument("--dropout",      type=float, default=0.1)
    g.add_argument("--num_paths",    type=int,   default=8)
    g.add_argument("--max_path_len", type=int,   default=3)
    g.add_argument("--num_negative", type=int,   default=256)
    g.add_argument("--max_history",  type=int,   default=64)

    # ── Training ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Training parametrlari")
    g.add_argument("--epochs",          type=int,   default=50,  dest="num_epochs")
    g.add_argument("--batch_size",      type=int,   default=512)
    g.add_argument("--lr",              type=float, default=3e-4, dest="learning_rate")
    g.add_argument("--weight_decay",    type=float, default=1e-4)
    g.add_argument("--grad_clip",       type=float, default=1.0)
    g.add_argument("--label_smoothing", type=float, default=0.1)
    g.add_argument("--w_link",          type=float, default=1.0)
    g.add_argument("--w_self_adv",      type=float, default=0.5)
    g.add_argument("--w_direct",        type=float, default=1.0)
    g.add_argument("--w_ortho_reg",     type=float, default=0.001)
    g.add_argument("--eval_every",      type=int,   default=1)

    # ── System ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Tizim")
    g.add_argument("--device",      default="cuda")
    g.add_argument("--seed",        type=int, default=42)
    g.add_argument("--num_workers", type=int, default=-1,
                   help="DataLoader workers (-1 = avtomatik)")
    g.add_argument("--save_dir",    default="checkpoints")
    g.add_argument("--log_dir",     default="logs")
    g.add_argument("--resume",      default=None,
                   help="Checkpoint fayldan davom ettirish")
    g.add_argument("--no_fp16",     action="store_true",
                   help="FP16 ni o'chirish")

    return p


def parse_config() -> Config:
    p   = build_parser()
    args = p.parse_args()

    cfg = Config()

    # Qaysi argumentlar CLI dan berilganini aniqlash
    defaults = {a.dest: p.get_default(a.dest) for a in p._actions}
    cli_set  = {k for k, v in vars(args).items()
                if v != defaults.get(k) and v is not None and k != "dataset"}

    # Argumentlarni cfg ga ko'chirish
    for k, v in vars(args).items():
        if k == "no_fp16":
            cfg.fp16 = not v
        elif k == "dataset":
            pass  # choose_dataset() hal qiladi
        elif hasattr(cfg, k):
            setattr(cfg, k, v)

    cfg._cli_args   = cli_set
    cfg._raw_dataset = args.dataset   # None yoki string
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  Asosiy funksiya
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg    = parse_config()
    logger = get_logger("orion", cfg.log_dir)

    # ── Dataset tanlash ───────────────────────────────────────────────────────
    cfg.dataset = choose_dataset(cfg.data_dir, cfg._raw_dataset)
    seed_everything(cfg.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        cfg.device = "cpu"
        cfg.fp16   = False
        logger.warning("GPU topilmadi — CPU ishlatiladi (sekin, GPU tavsiya etiladi)")
    else:
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu}  ({mem:.1f} GB)  |  CUDA {torch.version.cuda}")
        cfg.device = "cuda"

    # ── num_workers: Linux da fork → ko'p worker ──────────────────────────────
    cli = getattr(cfg, "_cli_args", set())
    if cfg.num_workers == -1 or "num_workers" not in cli:
        if platform.system() == "Linux":
            cfg.num_workers = min(8, os.cpu_count() or 4)
        else:
            cfg.num_workers = 0     # Windows: spawn og'ir
    logger.info(
        f"Device: {cfg.device}  |  FP16: {cfg.fp16}"
        f"  |  Workers: {cfg.num_workers}"
    )

    # ── Dataset-specific sozlamalar ───────────────────────────────────────────
    apply_dataset_config(cfg, cli)
    logger.info(
        f"Dataset: {cfg.dataset}  |  epochs={cfg.num_epochs}"
        f"  |  batch={cfg.batch_size}  |  lr={cfg.learning_rate}"
        f"  |  history={cfg.max_history}  |  paths={cfg.num_paths}"
    )

    # ── DataModule ────────────────────────────────────────────────────────────
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir,  exist_ok=True)

    dm = TKGDataModule(cfg)
    dm.setup()
    logger.info(
        f"|E|={cfg.num_entities}  |R|={cfg.num_relations}  |T|={cfg.num_times}"
        f"  |  Train:{len(dm.train_ds):,}"
        f"  Valid:{len(dm.valid_ds):,}"
        f"  Test:{len(dm.test_ds):,}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EliteTKGModel(
        num_entities       = cfg.num_entities,
        num_relations      = cfg.num_relations,
        num_times          = cfg.num_times,
        entity_dim         = cfg.entity_dim,
        relation_dim       = cfg.relation_dim,
        delta_dim          = cfg.delta_dim,
        hidden_dim         = cfg.hidden_dim,
        num_heads          = cfg.num_heads,
        num_layers         = cfg.num_layers,
        ffn_dim            = cfg.ffn_dim,
        num_negative       = cfg.num_negative,
        num_patterns       = cfg.num_patterns,
        dropout            = cfg.dropout,
        label_smoothing    = cfg.label_smoothing,
        w_direct           = cfg.w_direct,
        w_pattern_div      = cfg.w_pattern_div,
        w_copy             = cfg.w_copy,
        w_hist_contrast    = cfg.w_hist_contrast,
        use_direct_scoring = cfg.use_direct_scoring,
        use_diachronic     = cfg.use_diachronic,
        use_history        = cfg.use_history,
        max_history        = cfg.max_history,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parametrlari: {n_params/1e6:.2f}M")

    # ── Entity Frequency — HistoricalCopyHead normalization ───────────────────
    # HTKGP dan g'oya: hub entitylar (juda ko'p uchraydigan) ni bosish
    # → WIKI/YAGO da H@1 oshadi
    if model.copy_head is not None:
        entity_freq = dm.get_entity_freq()
        model.copy_head.set_entity_freq(entity_freq)
        logger.info(
            f"Entity freq yuklandi: max={entity_freq.max():.0f} "
            f"mean={entity_freq.mean():.1f} "
            f"(hub suppression yoqildi)"
        )

    # Multi-GPU (DataParallel)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        logger.info(f"DataParallel: {n_gpus} ta GPU")

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
    logger.info(f"Yakuniy test natijalari: {test_metrics}")


if __name__ == "__main__":
    main()
