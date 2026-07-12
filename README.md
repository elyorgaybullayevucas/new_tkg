# TREA-TKG — Temporal Knowledge Graph Link Prediction

**T**emporal **R**ecurrence-**E**nhanced **A**ttention for TKG forecasting.

Predicts the missing object entity for a query `(subject, relation, ?, time)`
using only facts strictly before `time` (extrapolation setting).

---

## Arxitektura

```
Input: (subject s, relation r, query_time t) + s ning tarixi: [(rel, obj, Δt), ...]
         |
         v
[1] EMBEDDING
    ent_emb[s], rel_emb[r]  (Xavier init, inverse relatsiyalar uchun 2R slot)
    query = h_s + h_r
         |
         v
[2] ADAPTIVE TEMPORAL ATTENTION  ★ yagona ilmiy yangilik
    Multi-head attention, s ning tarixi ustida.
    Har bir RELATION o'zining o'rganiluvchi vaqt-parchash (decay) tezligiga ega:
      attn_logit = (q·k)/√d − softplus(α_r)·Δt
    (DaeMon/CyGNet dagi qattiq/global decay'dan farqli — masalan "harbiy
    hujum" tez eskiradi, "ontologik" relation (YAGO) deyarli o'zgarmaydi)
         |
         v
[3] FUSION FFN
    query_repr = FFN([query, tarix konteksti])
         |
         v
    ┌────────────────────────┐        ┌──────────────────────────────┐
    │ A: DistMult scoring    │        │ B: Copy score                │
    │ (query_repr ⊙ h_r)·h_o │        │ recency × log(1+chastota),   │
    │                        │        │ trea/data.py da oldindan     │
    │                        │        │ hisoblanadi (O(occurrences)  │
    │                        │        │ per query, dataset-wide scan │
    │                        │        │ emas)                        │
    └───────────┬────────────┘        └──────────────┬───────────────┘
                └───────────────┬───────────────────--┘
                                v
[4] ADAPTIVE GATE
    g = σ(MLP(query_repr))
    final_logits = g·A + (1−g)·B

LOSS = label-smoothed CE (1-vs-N)  +  α · hard-negative triplet loss (batch ichida)
```

To'liq izoh: [trea/model.py](trea/model.py) docstringlarida.

---

## O'rnatish (Linux GPU server)

```bash
conda create -n trea python=3.10 -y
conda activate trea

# CUDA 12.x bo'lsa:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install tqdm numpy

# Data
unzip data.zip            # data/ papkasi hosil bo'ladi
ls data/                  # ICEWS18/  WIKI/  YAGO/  GDELT/
```

`data/ICEWS14/` repoda **yo'q qilingan** — u ICEWS18 ma'lumotlarining
nusxasi edi (haqiqiy ICEWS14 emas). Haqiqiy manba topilsa qayta qo'shiladi.

---

## Ishlatish

Har bir dataset uchun batch_size / history_len / early-stopping patience /
max_epochs avtomatik qo'llanadi (`trea/config.py:apply_dataset_defaults`) —
training validation MRR to'xtaganda avtomatik to'xtaydi, qat'iy epoch
sonida emas.

```bash
python train_trea.py --dataset ICEWS18
python train_trea.py --dataset WIKI
python train_trea.py --dataset YAGO
python train_trea.py --dataset GDELT

# yoki:
./run.sh ICEWS18

# Har qanday parametrni qo'lda override qilish
python train_trea.py --dataset YAGO --embed_dim 128 --lr 2e-3

# CPU (GPU yo'q bo'lsa)
python train_trea.py --dataset YAGO --device cpu --batch_size 64 --num_workers 0

# CPU-only sanity/smoke test (haqiqiy GPU kerak emas)
python -m trea.smoke_test
```

---

## Natijalar

Haqiqiy MRR/Hits@K raqamlari GPU serverda to'liq training tugagach
`checkpoints/trea/<DATASET>_results.json` ga saqlanadi — bu README'da
tekshirilmagan raqamlar keltirilmaydi.

---

## Fayl tuzilmasi

```
.
├── train_trea.py            # Ishga tushirish nuqtasi
├── run.sh                   # Linux server uchun wrapper skript
├── trea/
│   ├── config.py            # TREAConfig + apply_dataset_defaults()
│   ├── data.py               # GraphIndex (tarix + copy-score indekslash),
│   │                          #   TKGDataset, collate_fn, TKGDataLoader
│   ├── model.py              # TemporalEncoding, AdaptiveTemporalAttention,
│   │                          #   AdaptiveGate, TREAModel (DistMult scoring)
│   ├── loss.py               # LabelSmoothingCE + HardNegativeTripletLoss
│   ├── trainer.py            # TREATrainer: early stopping + checkpoint
│   ├── evaluate.py           # Filtered MRR/Hits@K (utils/metrics.py orqali)
│   └── smoke_test.py         # CPU-only correctness test
├── data/
│   ├── ICEWS18/ WIKI/ YAGO/ GDELT/   # entity2id.txt, relation2id.txt, train/valid/test.txt
├── utils/
│   ├── logging.py
│   └── metrics.py            # compute_ranks(), ranks_to_metrics() — trea ham shundan foydalanadi
```

---

## Muhim parametrlar

| Parametr | Default | Tavsif |
|----------|---------|--------|
| `--embed_dim` | 256 | Entity/relation embedding o'lchami |
| `--history_len` | dataset-ga qarab | Entity tarixidan qancha TIMESTEP orqaga qaraladi |
| `--num_heads` | 4 | Adaptive Temporal Attention head soni |
| `--copy_lambda` | 0.5 | Copy score uchun vaqt-parchash koeffitsiyenti |
| `--max_epochs` | dataset-ga qarab | Xavfsizlik chegarasi — odatda early stop undan oldin ishlaydi |
| `--early_stop_patience` | dataset-ga qarab | Necha eval round MRR yaxshilanmasa to'xtaydi |
| `--lr` | 1e-3 | Learning rate (CosineAnnealingLR) |
