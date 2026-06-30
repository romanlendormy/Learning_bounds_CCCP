#!/bin/bash
# run_collect_ssp.sh — Génération des datasets SSP chance-constrained (grille T×N).
#
# Usage direct (login node ou nœud interactif) :
#   bash run_collect_ssp.sh
#
# Usage SLURM :
#   sbatch run_collect_ssp.sh

#SBATCH --account=def-qcappart
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --job-name=collect_ssp
#SBATCH --output=logs/collect_ssp_%j.out

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$REPO/data"
OUT_BASE="$DATA_DIR/ssp_collect"

# ── Grille de collecte ────────────────────────────────────────────────────────
N_EPISODES=5
T_VALUES=(10 15 20)
N_VALUES=(20 50 100)
BASE_SEED=42          # seed de T10 N20 ; incrément +1 par config pour diversité

# ── Environnement ─────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || true
VENV_PATH="$HOME/envs/learning_bounds"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERREUR : venv introuvable à $VENV_PATH" >&2; exit 1
fi

mkdir -p "$OUT_BASE" logs

# ── Collectes ─────────────────────────────────────────────────────────────────
seed=$BASE_SEED
for T in "${T_VALUES[@]}"; do
    for N in "${N_VALUES[@]}"; do
        name="T${T}_N${N}"
        out_dir="$OUT_BASE/$name"
        echo "── $name  (n=$N_EPISODES, seed=$seed) → $out_dir"
        cd "$DATA_DIR"
        python -m ssp_generate.generator \
            --config ssp_generate/config_default.yaml \
            --T "$T" \
            --N "$N" \
            --n "$N_EPISODES" \
            --seed "$seed" \
            --output "$out_dir"
        echo "   OK — $(ls "$out_dir")"
        seed=$((seed + 1))
    done
done

echo ""
echo "Collectes terminées dans $OUT_BASE :"
for d in "$OUT_BASE"/*/; do
    ep_count=$(grep -c "^ep_" "$d/manifest.txt" 2>/dev/null || echo "?")
    echo "  $(basename "$d")  →  $ep_count épisodes"
done
