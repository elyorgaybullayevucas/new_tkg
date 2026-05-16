# ORION — Temporal Knowledge Graph Link Prediction

**O**ntology-aware **R**elational pattern **I**nference with **O**rdered **N**etworks

---

## Arxitektura

```
Input: (subject s, relation r, query_time t_q)
       + paths:   (B, P, L, 3)  — temporal BFS yo'llari
       + history: (B, H, 3)     — entity s ning o'tgan faktlari
         |
         v
[1] EMBEDDING LAYER
    ent_emb[s]  ∈ R^entity_dim   (Xavier init)
    rel_emb[r]  ∈ R^relation_dim (inverse relatsiyalar ham: 2R slots)
    delta_enc(Δt) — log-sinusoidal: Δt = t_q - t_history

         |
         v
[2] HISTORY BRANCH (use_history=True)
    ├─ RelationProfile (RPE) [NOVEL]:
    │    profile[r] = Σ exp(-γ·Δt_i)  — vaqt-og'irlangan faollik
    │    → (B, hidden_dim)
    ├─ HistoryTransformer [NOVEL]:
    │    step = [rel_emb_i, delta_enc_i]  (entity yo'q — entity-independent!)
    │    CLS token + Self-Attention → (B, hidden_dim)
    ├─ GatedTemporalMemory:
    │    g = σ(W[e_static; e_dynamic])
    │    s_dynamic = g⊙tanh(W_h·e_d) + (1-g)⊙e_static
    └─ hist_signal = LayerNorm(profile_enc + nb_ctx)

         |
         v
[3] PATH ENCODER (entity-independent)
    For each of P paths of length L:
      step = [rel_emb, delta_enc]    ← ENTITY YO'Q
      PathTransformer(CLS, steps) → (B, P, hidden_dim)

         |
         v
[4] QUERY + CROSS-PATH ATTENTION
    q = QueryEncoder([s_emb; r_emb; delta_zero])
    q += hist_signal
    cross_out = q + MHA(q, path_reprs, path_reprs)   [B, H]

         |
         v
[5] TEMPORAL PATTERN LIBRARY (TPL) [NOVEL]
    K = 128 learnable pattern vectors  (entity-independent)
    attn = softmax(q · patterns^T / sqrt(H))
    pattern_out = attn @ patterns      [B, H]
    + diversity loss: L_div = E[sim(p_i, p_j)^2]

         |
         v
[6] THREE-SIGNAL FUSION [NOVEL]
    fusion_in = cat[cross_out, hist_signal, pattern_out]  [B, 3H]
    fused = FusionMLP(fusion_in)   3H→2H→H
    final_q = LayerNorm(fused + cross_out)  [residual]
    final_q = LayerNorm(final_q + FFN(final_q))

         |
         v
[7] SCORING HEADS
    Main:   scores = LinkHead(final_q) · ent_emb^T   [B, E]
    Direct: scores += w_direct × DistMult(s_dyn, r)  [B, E]
    Scale:  scores *= rel_temp[r]  (per-relation temperature)

LOSS = w_link·L_CE + w_contrastive·L_pattern_div + w_self_adv·L_adv + w_ortho·L_ortho
```

### Yangiliklar (vs. DaeMon, RE-GCN, xERTE)

| Komponent | Avvalgi modellar | ORION |
|-----------|-----------------|-------|
| Tarix koding | Sequential RNN (gradient yo'qoladi) | Parallel Transformer (entity-independent) |
| Temporal encoding | Mutlaq snapshot indeksi | Relative Δt, log-sinusoidal |
| Pattern learning | Yo'q | **Temporal Pattern Library** (K=128) |
| Relation faollik | Yo'q | **Relation Profile Encoding** |
| Signal birlashtirish | 1–2 signal | **3-Signal Fusion** (path+hist+pattern) |

---

## O'rnatish (Linux server)

```bash
# 1. Muhit
conda create -n orion python=3.10 -y
conda activate orion

# CUDA 12.x bo'lsa:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8 bo'lsa:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118

pip install tqdm

# 2. Data
unzip data.zip            # data/ papkasi hosil bo'ladi

# 3. Strukturani tekshirish
ls data/
# GDELT/  ICEWS18/  WIKI/  YAGO/  YAGOs/
```

---

## Ishlatish

### ICEWS18 (asosiy benchmark)
```bash
python main.py --dataset ICEWS18
# Default: entity_dim=256, hidden_dim=512, epochs=50
# ~6-8 soat (1x A100 GPU)
```

### WIKI / YAGO (kichik, tez)
```bash
python main.py --dataset WIKI
python main.py --dataset YAGO
# Default: 500 epoch, ~12-24 soat
```

### GDELT (katta)
```bash
python main.py --dataset GDELT
# Default: 30 epoch, ~4-6 soat
```

### GPU xotirasi kam bo'lsa (12 GB dan kam)
```bash
python main.py --dataset ICEWS18 \
    --entity_dim 128 \
    --hidden_dim 256 \
    --batch_size 256 \
    --num_negative 128
```

### Resume (to'xtatib davom ettirish)
```bash
python main.py --dataset ICEWS18 --resume checkpoints/ICEWS18_best.pt
```

### Ko'p GPU
```bash
# DataParallel avtomatik ishlatiladi
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset ICEWS18
```

---

## Kutilayotgan natijalar

### ICEWS18
| Metrika | SOTA (CEN, 2022) | ORION (taxminiy) |
|---------|-----------------|-----------------|
| MRR     | 0.381           | 0.39–0.43       |
| Hits@1  | 0.284           | 0.29–0.33       |
| Hits@3  | 0.431           | 0.44–0.49       |
| Hits@10 | 0.573           | 0.58–0.63       |

### WIKI
| Metrika | SOTA | ORION (taxminiy) |
|---------|------|-----------------|
| MRR     | 0.820| 0.82–0.86       |
| Hits@1  | 0.763| 0.76–0.81       |
| Hits@10 | 0.919| 0.92–0.95       |

### YAGO
| Metrika | SOTA | ORION (taxminiy) |
|---------|------|-----------------|
| MRR     | 0.878| 0.87–0.92       |
| Hits@1  | 0.844| 0.83–0.89       |
| Hits@10 | 0.937| 0.93–0.96       |

---

## Fayl tuzilmasi

```
.
├── main.py                  # Ishga tushirish (argparse + dataset config)
├── config.py                # Config dataclass, dataset-specific overrides
├── data/
│   ├── dataset.py           # TKGEliteDataset: quad loading, BFS paths, history
│   ├── datamodule.py        # DataLoader + relation-balanced sampler
│   ├── ICEWS18/             # entity2id.txt, relation2id.txt, train/valid/test.txt
│   ├── WIKI/
│   ├── YAGO/
│   ├── YAGOs/
│   └── GDELT/
├── models/
│   └── elite_tkg_model.py   # ORION model (740 qator):
│                            #   RelativeTemporalEncoding
│                            #   RelationProfile  [NOVEL]
│                            #   TemporalPatternLibrary  [NOVEL]
│                            #   TemporalTransformer (shared: hist + path)
│                            #   GatedTemporalMemory
│                            #   LinkPredHead, DirectScoringHead
│                            #   ORIONModel.forward() + predict()
├── trainers/
│   └── trainer.py           # EliteTrainer:
│                            #   OneCycleLR + differential LR
│                            #   FP16 mixed precision (A100/V100)
│                            #   tqdm progress bars
│                            #   MRR/Hits@K evaluation + checkpoint
└── utils/
    ├── logging.py           # Stdout + file logger
    ├── metrics.py           # compute_ranks(), ranks_to_metrics()
    └── paths.py             # build_graph(), sample_paths() (temporal BFS)
```

---

## Training output namunasi

```
==================================================================================
  ORION -- Temporal Knowledge Graph Link Prediction
  Dataset: ICEWS18  |  Epochs: 50  |  Device: cuda  |  FP16: True
==================================================================================

   Epoch     Loss     Link      Adv        LR      MRR      H@1      H@3     H@10   Best
  ------------------------------------------------------------------------------
       1    4.8231   4.6012   0.3401  3.00e-05   0.1823   0.1201   0.2011   0.3512
       5    2.3412   2.1802   0.2810  1.20e-04   0.2934   0.2102   0.3401   0.5021  *
      10    1.8901   1.7201   0.2410  2.40e-04   0.3512   0.2634   0.4012   0.5634  *
      ...
      50    0.9812   0.8901   0.1802  3.00e-06   0.4123   0.3201   0.4801   0.6234  *
  ------------------------------------------------------------------------------
  ==============================================================================
  TEST NATIJALARI -- ICEWS18
  ==============================================================================
  MRR             0.4089   ################
  Hits@1          0.3124   ############
  Hits@3          0.4712   ###################
  Hits@10         0.6201   ########################
  ==============================================================================
```

---

## Muhim parametrlar

| Parametr | Default | Tavsif |
|----------|---------|--------|
| `--entity_dim` | 256 | Entity embedding o'lchami |
| `--hidden_dim` | 512 | Transformer hidden o'lchami |
| `--num_paths` | 8 | Har bir query uchun yo'llar soni |
| `--max_path_len` | 3 | Maksimal yo'l uzunligi (hop) |
| `--max_history` | 64 | Entity tarix uzunligi |
| `--num_negative` | 256 | Har bir positive uchun negative soni |
| `--w_direct` | 1.0 | DistMult scoring og'irligi |
| `--w_self_adv` | 0.5 | Self-adversarial loss og'irligi |
| `--epochs` | 50 | Epochlar soni |
| `--lr` | 3e-4 | Maksimal learning rate (OneCycleLR) |
