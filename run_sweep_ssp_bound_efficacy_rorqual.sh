#!/bin/bash
# run_sweep_ssp_bound_efficacy_rorqual.sh — Sweep (E,T,N) efficacité borne v1CLD sur SSP.
#
# Compare, sur des instances calibrées par le générateur adaptatif
# (ssp_stochastic.generator_adaptive), B&B base_ref (même B&B custom, SANS
# borne) vs B&B v1CLD (même B&B, AVEC borne glouton) — cf.
# ssp_stochastic/sweep_bound_efficacy.py. v1LD désactivé (coûteux, hors scope).
#
# Grille par défaut : (E,T)∈{(3,12),(4,8),(4,12)} × N∈{20,50} × 3 reps = 18
# instances — seules ces 3 paires (E,T) atteignent la fenêtre de difficulté
# cible [1e4,1e6] nœuds sous Gecode base (cf. étude de calibration : E∈{2,3,4}
# avec T≤8, ou E=4 avec T=5, basculent directement de trivial à INFEASIBLE
# sans palier dur intermédiaire, même à N=50).
#
# Soumettre :
#   sbatch run_sweep_ssp_bound_efficacy_rorqual.sh
#
# Scope minimal (smoke test avant le sweep complet) :
#   sbatch --export=ALL,COMBOS="3:12",N_VALUES="20",N_REPS=1 run_sweep_ssp_bound_efficacy_rorqual.sh
#
# Varier le budget :
#   sbatch --export=ALL,TIME_LIM=120 run_sweep_ssp_bound_efficacy_rorqual.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=sweep_ssp_bound_efficacy
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres (surchargeables via --export=ALL,...) ──────────────────────────
COMBOS="${COMBOS:-3:12 4:8 4:12}"   # séparateur ':' (pas ',', réservé par --export de sbatch)
N_VALUES="${N_VALUES:-20 50}"
EPS="${EPS:-0.10}"
N_REPS="${N_REPS:-3}"
T_NODE="${T_NODE:-30}"
TIME_LIM="${TIME_LIM:-300}"         # secondes, par méthode (base_ref, v1CLD) — v1CLD
                                     # a un coût par nœud élevé sur SSP (cf. smoke test :
                                     # ~24ms/nœud, borne=99.85% du temps), 60s était trop
                                     # court pour le laisser dépasser base_ref en nœuds
PROBE_TIME_LIM="${PROBE_TIME_LIM:-5.0}"
TARGET_LO="${TARGET_LO:-1e4}"
TARGET_HI="${TARGET_HI:-1e6}"
MASTER_SEED="${MASTER_SEED:-42}"

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
BENCH_BIN="$REPO/solver_bab/build/bench_ssp_bab_file"
PROBE_BIN="$REPO/solver_bab/build/ssp_probe"
VENV_PATH="$HOME/envs/learning_bounds"
OUT_CSV="$REPO/ssp_stochastic/results/sweep_bound_efficacy_${SLURM_JOB_ID:-local}.csv"

# ── Modules & venv ──────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3.0 python/3.11 2>/dev/null || true
source "$VENV_PATH/bin/activate"

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$REPO/logs" "$REPO/ssp_stochastic/results"

echo "======================================================================"
echo "  sweep_ssp_bound_efficacy — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  combos(E,T) : $COMBOS   N : $N_VALUES   eps=$EPS   n_reps=$N_REPS"
echo "  T_node=$T_NODE   time_lim=${TIME_LIM}s/méthode   target=[$TARGET_LO,$TARGET_HI] nœuds"
echo "  CSV : $OUT_CSV"
echo "======================================================================"

# ── Vérification des binaires ─────────────────────────────────────────────────
for BIN in "$BENCH_BIN" "$PROBE_BIN"; do
    if [ ! -x "$BIN" ]; then
        echo "ERREUR : binaire introuvable : $BIN"
        echo "  → Compiler (build incrémental, PAS compile_cc.sh qui ne reconstruit"
        echo "    que les cibles MKP) :"
        echo "       cmake --build $REPO/solver_bab/build --target bench_ssp_bab_file ssp_probe"
        exit 1
    fi
done

# ── Run ────────────────────────────────────────────────────────────────────────
echo ""
python -m ssp_stochastic.sweep_bound_efficacy \
    --combos $COMBOS \
    --N-values $N_VALUES \
    --eps "$EPS" \
    --n-reps "$N_REPS" \
    --T-node "$T_NODE" \
    --time-lim "$TIME_LIM" \
    --probe-time-lim "$PROBE_TIME_LIM" \
    --target-lo "$TARGET_LO" \
    --target-hi "$TARGET_HI" \
    --master-seed "$MASTER_SEED" \
    --bench-bin "$BENCH_BIN" \
    --probe-bin "$PROBE_BIN" \
    --out "$OUT_CSV"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  CSV : $OUT_CSV"
echo "======================================================================"
