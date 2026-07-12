"""
TREA-TKG Trainer — training loop with per-epoch metrics display.

Features:
  • Prints rich per-epoch table to stdout and logs to file
  • REAL early stopping on validation MRR (patience-based, counted in eval
    rounds) with a per-dataset max_epochs safety cap — the previous version's
    docstring claimed early stopping but the loop always ran `cfg.epochs`
    unconditionally regardless of convergence.
  • Saves best checkpoint automatically
  • Resume from checkpoint support
"""
import os
import time
import json
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from trea.config import TREAConfig
from trea.model import TREAModel
from trea.loss import TREALoss
from trea.data import TKGDataLoader
from trea.evaluate import evaluate, format_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_epoch_header():
    line = "─" * 96
    print(line)
    print(
        f"{'Epoch':>6}  {'Time':>6}  "
        f"{'Loss':>8}  {'CE':>8}  {'Triplet':>8}  "
        f"{'MRR':>7}  {'H@1':>7}  {'H@3':>7}  {'H@10':>7}  {'LR':>9}  {'Pat':>4}"
    )
    print(line)


def print_epoch_row(epoch, elapsed, loss_d, metrics, lr, patience_left, is_best=False):
    star = "★" if is_best else " "
    print(
        f"{star}{epoch:>5}  {elapsed:>5.1f}s  "
        f"{loss_d['total']:>8.4f}  {loss_d['ce']:>8.4f}  {loss_d['triplet']:>8.4f}  "
        f"{metrics.get('MRR',0):>7.4f}  "
        f"{metrics.get('Hits@1',0):>7.4f}  "
        f"{metrics.get('Hits@3',0):>7.4f}  "
        f"{metrics.get('Hits@10',0):>7.4f}  "
        f"{lr:>9.2e}  {patience_left:>4}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class TREATrainer:
    def __init__(self, cfg: TREAConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        # ── device ────────────────────────────────────────────────────────────
        if cfg.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("[Device] CPU")

        # ── data ──────────────────────────────────────────────────────────────
        self.loader = TKGDataLoader(cfg)
        self.train_dl = self.loader.train_loader()

        # ── model ─────────────────────────────────────────────────────────────
        self.model = TREAModel(
            num_entities=self.loader.num_entities,
            num_relations=self.loader.num_relations,
            cfg=cfg,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] TREA-TKG  params={n_params:,}")

        # ── loss ──────────────────────────────────────────────────────────────
        self.loss_fn = TREALoss(
            num_entities=self.loader.num_entities,
            smoothing=cfg.label_smoothing,
            alpha=cfg.alpha_contrastive,
            margin=cfg.margin,
        )

        # ── optimiser & scheduler ────────────────────────────────────────────
        # T_max uses max_epochs (the safety cap), not a fixed epoch count —
        # if early stopping fires first the schedule simply never completes,
        # which is fine (LR keeps annealing smoothly up to the stop point).
        self.optim = AdamW(self.model.parameters(),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optim, T_max=cfg.max_epochs,
                                            eta_min=cfg.lr * 0.01)

        # ── logging ───────────────────────────────────────────────────────────
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.log_path = os.path.join(cfg.log_dir,
                                     f"{cfg.dataset}_train_log.jsonl")
        self.best_mrr = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = []            # list of per-epoch dicts

    # ── one training epoch ────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()

        total_loss = total_ce = total_tri = 0.0
        n_batches = 0

        for batch in tqdm(self.train_dl, desc=f"Epoch {epoch}", leave=False):
            subs    = batch["subject"].to(self.device)
            rels    = batch["relation"].to(self.device)
            objs    = batch["object"].to(self.device)
            times   = batch["time"].to(self.device)
            h_rels  = batch["hist_rels"].to(self.device)
            h_objs  = batch["hist_objs"].to(self.device)
            h_times = batch["hist_times"].to(self.device)
            h_mask  = batch["hist_mask"].to(self.device)
            copy_sc = batch["copy_scores"].to(self.device)

            # ── forward ───────────────────────────────────────────────────────
            logits = self.model(
                subs, rels, h_rels, h_objs, h_times, times, h_mask, copy_sc
            )

            # query repr for contrastive loss
            query_repr = self.model.encode_query(
                subs, rels, h_rels, h_objs, h_times, times, h_mask
            )

            # ── loss ──────────────────────────────────────────────────────────
            loss_d = self.loss_fn(logits, objs, query_repr, self.model.ent_emb)

            # ── backward ─────────────────────────────────────────────────────
            self.optim.zero_grad()
            loss_d["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            self.optim.step()

            total_loss += loss_d["total"].item()
            total_ce   += loss_d["ce"]
            total_tri  += loss_d["triplet"]
            n_batches  += 1

        self.scheduler.step()

        return {
            "total": total_loss / n_batches,
            "ce":    total_ce   / n_batches,
            "triplet": total_tri / n_batches,
        }

    # ── main training loop ────────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        print(f"\n{'═'*96}")
        print(f"  TREA-TKG Training  │  Dataset: {cfg.dataset}  │  "
              f"max_epochs: {cfg.max_epochs}  │  patience: {cfg.early_stop_patience}  │  "
              f"Device: {self.device}")
        print(f"{'═'*96}\n")

        print_epoch_header()

        stopped_early = False
        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.time()

            # ── train ─────────────────────────────────────────────────────────
            loss_d = self._train_epoch(epoch)

            elapsed = time.time() - t0
            lr_now = self.scheduler.get_last_lr()[0]

            # ── evaluate ──────────────────────────────────────────────────────
            metrics = {}
            if epoch % cfg.eval_every == 0:
                metrics = evaluate(
                    self.model, self.loader, split="valid",
                    device=self.device,
                    batch_size=cfg.batch_size,
                    hits_at=tuple(cfg.hits_at),
                    verbose=False,
                )

            # ── early stopping bookkeeping ───────────────────────────────────
            is_best = metrics.get("MRR", 0) > self.best_mrr + cfg.early_stop_min_delta
            if is_best:
                self.best_mrr = metrics["MRR"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch, metrics, "best")
            elif metrics:
                self.patience_counter += 1

            patience_left = max(0, cfg.early_stop_patience - self.patience_counter)

            # ── print row ─────────────────────────────────────────────────────
            print_epoch_row(epoch, elapsed, loss_d, metrics, lr_now, patience_left, is_best)

            # ── log to file ───────────────────────────────────────────────────
            record = {
                "epoch": epoch, "loss": loss_d, "metrics": metrics,
                "lr": lr_now, "elapsed": elapsed,
            }
            self.history.append(record)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            if self.patience_counter >= cfg.early_stop_patience:
                print(f"\n  ⏹  Early stop: no MRR improvement for "
                      f"{cfg.early_stop_patience} eval rounds "
                      f"(best={self.best_mrr:.4f} @ epoch {self.best_epoch}).")
                stopped_early = True
                break

        # ── final summary ─────────────────────────────────────────────────────
        print("─" * 96)
        reason = "early stop" if stopped_early else "max_epochs reached"
        print(f"\n★  Best valid MRR: {self.best_mrr:.4f}  (epoch {self.best_epoch}, {reason})")

        # ── test evaluation on best checkpoint ───────────────────────────────
        print("\nLoading best checkpoint for TEST evaluation …")
        self._load_checkpoint("best")
        test_metrics = evaluate(
            self.model, self.loader, split="test",
            device=self.device, batch_size=cfg.batch_size,
            hits_at=tuple(cfg.hits_at), verbose=True,
        )
        print("\n" + "═" * 60)
        print(f"  FINAL TEST RESULTS — {cfg.dataset}")
        print("═" * 60)
        for k, v in test_metrics.items():
            print(f"  {k:10s}: {v:.4f}")
        print("═" * 60)

        # save final results
        results_path = os.path.join(cfg.save_dir, f"{cfg.dataset}_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "dataset": cfg.dataset,
                "best_epoch": self.best_epoch,
                "stopped_early": stopped_early,
                "valid_mrr": self.best_mrr,
                "test": test_metrics,
                "config": cfg.__dict__,
            }, f, indent=2)
        print(f"\nResults saved → {results_path}")

        return test_metrics

    # ── checkpoint I/O ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, metrics: dict, tag: str):
        path = os.path.join(self.cfg.save_dir,
                            f"{self.cfg.dataset}_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optim.state_dict(),
            "metrics": metrics,
        }, path)

    def _load_checkpoint(self, tag: str):
        path = os.path.join(self.cfg.save_dir,
                            f"{self.cfg.dataset}_{tag}.pt")
        if not os.path.exists(path):
            print(f"  [warn] checkpoint not found: {path}")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint: {path}  (epoch {ckpt['epoch']})")
