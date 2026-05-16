#!/bin/bash
# ORION — Temporal KG Link Prediction
# Linux server da ishlatish uchun skript
#
# Ishlatish:
#   chmod +x run.sh
#   ./run.sh ICEWS18           # ICEWS18 dataset
#   ./run.sh WIKI              # WIKI dataset
#   ./run.sh YAGO              # YAGO dataset
#   ./run.sh GDELT             # GDELT dataset
#   ./run.sh ICEWS18 --resume checkpoints/ICEWS18_best.pt
#   ./run.sh ICEWS18 --entity_dim 128 --hidden_dim 256   # kichik GPU

set -e

DATASET=${1:-ICEWS18}
shift 2>/dev/null || true  # birinchi argumentdan keyin qolganlari extra args

echo "======================================================================"
echo "  ORION TKG — $DATASET"
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
mkdir -p checkpoints logs

# O'qitish
python main.py \
    --dataset "$DATASET" \
    "$@"

echo ""
echo "======================================================================"
echo "  Natijalar: checkpoints/${DATASET}_best.pt"
echo "  Loglar:    logs/"
echo "======================================================================"
