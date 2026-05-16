# trainers/trainer.py
"""
EliteTrainer — Mixed precision, OneCycleLR, tqdm progress bars.
"""
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

# PyTorch >= 2.0 yangi AMP API
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from config import Config
from data.dataset import TKGEliteDataset
from models.elite_tkg_model import EliteTKGModel
from utils.logging import get_logger
from utils.metrics import compute_ranks, format_metrics, ranks_to_metrics

# ── Ranglar ────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _bar(label: str, val: float, width: int = 12) -> str:
    filled = int(round(val * width))
    return f"{label}[{'█'*filled}{'░'*(width-filled)}] {val:.4f}"


class EliteTrainer:

    def __init__(
        self,
        model: EliteTKGModel,
        cfg: Config,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader:  DataLoader,
        valid_dataset: TKGEliteDataset,
        test_dataset:  TKGEliteDataset,
    ):
        self.model   = model   # may be nn.DataParallel
        self._raw    = model.module if hasattr(model, "module") else model  # raw model
        self.cfg     = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader  = test_loader
        self.valid_dataset = valid_dataset
        self.test_dataset  = test_dataset

        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        self._raw.to(self.device)

        self.logger = get_logger("elite_trainer", cfg.log_dir)

        # ── Optimizer ─────────────────────────────────────────────────────────
        # CATRE: ent_emb + rel_emb + delta_enc — sekin o'qitish (_raw dan olinadi)
        emb_params: list = []
        for name in ("ent_emb", "rel_emb", "delta_enc", "embeddings"):
            m = getattr(self._raw, name, None)
            if m is not None:
                emb_params += list(m.parameters())

        # Neighborhood + pattern + direct + diachronic — o'rta tezlik
        extra_params: list = []
        for name in ("relation_profile", "hist_transformer", "hist_to_entity",
                     "gate_mem", "nb_ctx", "hist_norm", "query_hist_norm", "pattern_lib",
                     "copy_head", "direct_head", "dia_amp", "dia_freq", "dia_phase",
                     # backward compat
                     "msa", "rel_mem", "pna", "csa", "history_encoder", "hist_gate"):
            m = getattr(self._raw, name, None)
            if m is not None:
                extra_params += list(m.parameters())

        # Qolgan barcha parametrlar — to'liq tezlik
        emb_set   = {id(p) for p in emb_params}
        extra_set = {id(p) for p in extra_params}
        other_params = [
            p for p in self._raw.parameters()
            if id(p) not in emb_set and id(p) not in extra_set
        ]
        self.optimizer = AdamW([
            {"params": emb_params,    "lr": cfg.learning_rate * 0.1},
            {"params": extra_params,  "lr": cfg.learning_rate * 0.5},
            {"params": other_params,  "lr": cfg.learning_rate},
        ], weight_decay=cfg.weight_decay)

        # ── Scheduler: OneCycleLR ─────────────────────────────────────────────
        steps_per_epoch = len(train_loader)
        total_steps = cfg.num_epochs * steps_per_epoch
        # param_groups soniga qarab max_lr ro'yxati
        base_lr  = cfg.learning_rate
        # 3 ta param group: emb, extra, other
        max_lrs = [base_lr * 0.1, base_lr * 0.5, base_lr]
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=0.05,          # warmup 5%
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e4,
        )

        # ── Mixed precision ───────────────────────────────────────────────────
        self.use_fp16 = cfg.fp16 and self.device.type == "cuda"
        if self.use_fp16:
            try:
                self.scaler = GradScaler(device="cuda")   # PyTorch >= 2.3
            except TypeError:
                self.scaler = GradScaler()                # eski versiya
        else:
            self.scaler = None

        self.best_mrr   = 0.0
        self.best_epoch = 0
        os.makedirs(cfg.save_dir, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────────

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {
            "loss": 0, "link": 0, "contrastive": 0, "self_adv": 0,
            "ortho": 0, "hist_contrast": 0
        }
        n = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"{CYAN}Epoch {epoch+1:>3}/{self.cfg.num_epochs}{RESET}",
            unit="batch",
            ncols=110,
            leave=False,
        )
        for batch in pbar:
            subjects    = batch["subject"].to(self.device)
            relations   = batch["relation"].to(self.device)
            objects     = batch["object"].to(self.device)
            times       = batch["time"].to(self.device)
            paths       = batch["paths"].to(self.device)
            path_masks  = batch["path_masks"].to(self.device)
            neg_objects = batch["neg_objects"].to(self.device)
            history     = batch["history"].to(self.device)
            hist_mask   = batch["hist_mask"].to(self.device)

            with autocast(device_type="cuda", enabled=self.use_fp16):
                scores, losses = self.model(
                    subjects, relations, objects, times,
                    paths, path_masks, neg_objects,
                    history=history, hist_mask=hist_mask,
                )
                total_loss = (
                    self.cfg.w_link          * losses["link"]
                    + self.cfg.w_contrastive * losses["contrastive"]
                    + self.cfg.w_self_adv    * losses["self_adv"]
                    + self.cfg.w_ortho_reg   * losses["ortho_reg"]
                    + self.cfg.w_hist_contrast * losses.get(
                        "hist_contrast", torch.tensor(0.0, device=subjects.device))
                )

            self.optimizer.zero_grad()
            if self.use_fp16:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self._raw.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(self._raw.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            self.scheduler.step()

            totals["loss"]          += total_loss.item()
            totals["link"]          += losses["link"].item()
            totals["contrastive"]   += losses["contrastive"].item()
            totals["self_adv"]      += losses["self_adv"].item()
            totals["ortho"]         += losses["ortho_reg"].item()
            totals["hist_contrast"] += losses.get("hist_contrast",
                                           torch.tensor(0.0)).item()
            n += 1

            if n % max(1, len(self.train_loader) // 5) == 0:
                avg_loss = totals["loss"] / n
                avg_link = totals["link"] / n
                lr_cur   = self.optimizer.param_groups[-1]["lr"]
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "link": f"{avg_link:.4f}",
                    "lr":   f"{lr_cur:.1e}",
                })

        pbar.close()
        lr = self.optimizer.param_groups[-1]["lr"]
        return {k: v / max(n, 1) for k, v in totals.items()} | {"lr": lr}

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, dataset: TKGEliteDataset,
                 desc: str = "Eval") -> Dict[str, float]:
        self.model.eval()
        all_ranks: List[torch.Tensor] = []

        pbar = tqdm(loader, desc=f"  {YELLOW}{desc}{RESET}", unit="batch",
                    ncols=90, leave=False)
        for batch in pbar:
            subjects    = batch["subject"].to(self.device)
            relations   = batch["relation"].to(self.device)
            objects     = batch["object"].to(self.device)
            times       = batch["time"].to(self.device)
            paths       = batch["paths"].to(self.device)
            path_masks  = batch["path_masks"].to(self.device)
            history     = batch["history"].to(self.device)
            hist_mask   = batch["hist_mask"].to(self.device)

            with autocast(device_type="cuda", enabled=self.use_fp16):
                scores = self._raw.predict(
                    subjects, relations, times, paths, path_masks,
                    history=history, hist_mask=hist_mask,
                )

            filter_mask = dataset.get_filter_mask(
                subjects.cpu(), relations.cpu(),
                times.cpu(), objects.cpu(),
            ).to(self.device)

            ranks = compute_ranks(
                scores, objects, filter_mask,
                filter_flag=self.cfg.filter_flag,
            )
            all_ranks.append(ranks.float().cpu())

        pbar.close()
        all_ranks_t = torch.cat(all_ranks)
        return ranks_to_metrics(all_ranks_t, self.cfg.hits_at_k)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, epoch: int, metrics: Dict, tag: str = "best"):
        path = os.path.join(self.cfg.save_dir, f"{self.cfg.dataset}_{tag}.pt")
        torch.save({
            "epoch":       epoch,
            "model_state": self._raw.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "sched_state": self.scheduler.state_dict(),
            "metrics":     metrics,
            "config":      self.cfg.__dict__,
        }, path)
        self.logger.info(f"  → Checkpoint: {path}")

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self._raw.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.logger.info(f"Yuklandi: {path} (epoch {ckpt['epoch']})")
        return ckpt["epoch"]

    # ── To'liq o'qitish ───────────────────────────────────────────────────────

    def _print_header(self):
        w = 82
        sep = "=" * w
        print(f"\n{BOLD}{sep}{RESET}")
        print(f"{BOLD}  ORION -- Temporal Knowledge Graph Link Prediction{RESET}")
        print(f"  Dataset: {CYAN}{self.cfg.dataset}{RESET}  |  "
              f"Epochs: {self.cfg.num_epochs}  |  "
              f"Device: {self.cfg.device}  |  FP16: {self.use_fp16}")
        print(f"{BOLD}{sep}{RESET}\n")
        # Column header
        print(f"  {'Epoch':>6}  {'Loss':>7}  {'Link':>7}  "
              f"{'Adv':>7}  {'LR':>8}  "
              f"{'MRR':>7}  {'H@1':>7}  {'H@3':>7}  {'H@10':>7}  {'Best':>5}")
        print(f"  {'-'*78}")

    def _print_row(self, epoch: int, tr: Dict, vm: Optional[Dict], is_best: bool):
        mrr  = f"{vm['MRR']:.4f}"   if vm else "  —   "
        h1   = f"{vm['Hits@1']:.4f}" if vm else "  —   "
        h3   = f"{vm['Hits@3']:.4f}" if vm else "  —   "
        h10  = f"{vm['Hits@10']:.4f}" if vm else "  —   "
        star = f"{GREEN}★{RESET}" if is_best else " "
        print(f"  {epoch:>6}  {tr['loss']:>7.4f}  {tr['link']:>7.4f}  "
              f"{tr['self_adv']:>7.4f}  {tr['lr']:>8.2e}  "
              f"{mrr:>7}  {h1:>7}  {h3:>7}  {h10:>7}  {star}")

    def fit(self) -> Dict[str, float]:
        start = 0
        if self.cfg.resume:
            start = self.load(self.cfg.resume) + 1

        self._print_header()
        self.logger.info(f"O'qitish boshlandi: {self.cfg.dataset}  epochs={self.cfg.num_epochs}")

        t0 = time.time()
        for epoch in range(start, self.cfg.num_epochs):
            t_ep = time.time()
            tr = self.train_one_epoch(epoch)

            vm       = None
            is_best  = False
            if (epoch + 1) % self.cfg.eval_every == 0:
                vm = self.evaluate(self.valid_loader, self.valid_dataset,
                                   desc=f"Valid e{epoch+1}")
                if vm["MRR"] > self.best_mrr:
                    self.best_mrr   = vm["MRR"]
                    self.best_epoch = epoch + 1
                    self.save(epoch + 1, vm, "best")
                    is_best = True

            self._print_row(epoch + 1, tr, vm, is_best)

            elapsed = time.time() - t_ep
            self.logger.info(
                f"Epoch {epoch+1}/{self.cfg.num_epochs} | "
                f"Loss:{tr['loss']:.4f} Link:{tr['link']:.4f} "
                f"Adv:{tr['self_adv']:.4f} LR:{tr['lr']:.2e} | "
                f"t={elapsed:.1f}s"
                + (f" | Valid: {format_metrics(vm)}" if vm else "")
            )

        # ── Test ──────────────────────────────────────────────────────────────
        total = time.time() - t0
        w = 82
        print(f"\n  {'-'*78}")
        print(f"  {BOLD}O'qitish yakunlandi{RESET}  |  "
              f"Jami: {total/60:.1f} min  |  "
              f"Eng yaxshi epoch: {self.best_epoch}  "
              f"Valid MRR: {GREEN}{self.best_mrr:.4f}{RESET}")

        best_path = os.path.join(self.cfg.save_dir, f"{self.cfg.dataset}_best.pt")
        if os.path.exists(best_path):
            self.load(best_path)

        tm = self.evaluate(self.test_loader, self.test_dataset, desc="Test")

        # ── Final metrics box ─────────────────────────────────────────────────
        print(f"\n{BOLD}{'='*w}{RESET}")
        print(f"{BOLD}  TEST NATIJALARI -- {self.cfg.dataset}{RESET}")
        print(f"{'='*w}")
        print(f"  {'Metrika':<14} {'Qiymat':>10}   {'Bar':}")
        print(f"  {'-'*50}")
        for key in ["MRR", "Hits@1", "Hits@3", "Hits@10"]:
            val = tm.get(key, 0)
            bar = "#" * int(val * 40)
            print(f"  {key:<14} {GREEN}{val:>10.4f}{RESET}   {bar}")
        print(f"{BOLD}{'='*w}{RESET}\n")

        self.logger.info(f"[TEST] {format_metrics(tm)}")
        return tm
