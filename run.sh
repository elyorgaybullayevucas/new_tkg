#!/bin/bash
# TREA-TKG — Temporal KG Link Prediction
# Linux server da ishlatish uchun skript
#
# Ishlatish:
#   chmod +x run.sh
#   ./run.sh ICEWS18           # ICEWS18 dataset
#   ./run.sh WIKI              # WIKI dataset
#   ./run.sh YAGO              # YAGO dataset
#   ./run.sh GDELT             # GDELT dataset
#   ./run.sh ICEWS18 --embed_dim 128   # kichik GPU

set -e

DATASET=${1:-ICEWS18}
shift 2>/dev/null || true  # birinchi argumentdan keyin qolganlari extra args

echo "======================================================================"
echo "  TREA-TKG — $DATASET"
echo "  $(date)"
echo "======================================================================"

# Muhitni tekshirish
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
    print(f'CUDA: {torch.version.cuda}')
else:
    print('WARNING: GPU topilmadi, CPU ishlatiladi (sekin!)')
"

# Checkpoint papkasi
mkdir -p checkpoints/trea logs/trea

# O'qitish — per-dataset default'lar (batch_size, history_len, early stopping
# patience, max_epochs) trea/config.py:apply_dataset_defaults() da avtomatik
# qo'llanadi; kerak bo'lsa "$@" orqali istalgan flag override qilinadi.
python train_trea.py \
    --dataset "$DATASET" \
    "$@"

echo ""
echo "======================================================================"
echo "  Natijalar: checkpoints/trea/${DATASET}_results.json"
echo "  Loglar:    logs/trea/"
echo "======================================================================"
