#!/bin/bash
# compare_ref_v1ld_v1cld_gnn_gen.sh — Comparaison à 5 méthodes de borne du B&B
# Gecode CCMKP (REF, V1LD, GNN, GNN_ROOT, V1CLD), même structure que
# compare_ref_v1ld_v1cld_gnn_n10.sh, mais sur UNE instance générée à la volée
# (même distribution que la collecte collect_s10 : generate_pisinger_instance
# + generate_scenario, cf. collect/collect_large_base_samples.py) avec les
# mêmes paramètres par défaut que compare_ref_gnn_song.sh (n=20/m=5/N=100),
# au lieu du fichier figé de Song et al. ou des fichiers pisinger pré-générés.
#
#   1. ref      : squelette B&B nu, sans borne (référence pour isoler l'effet
#                 du branchement/incumbent/moteur DFS de l'effet de la borne).
#   2. v1ld     : borne V1LD par descente de sous-gradient (T_root/T_node iters).
#   3. gnn      : borne V1LD one-shot par GNN, backend natif (torch::jit,
#                 device=cuda) — pas d'overhead IPC, recalculée à CHAQUE nœud.
#   4. gnn_root : même inférence GNN one-shot, mais UNE SEULE FOIS (à la
#                 racine) — REF pour tous les nœuds suivants (aucune borne
#                 recalculée). Teste si la borne de racine seule suffit à
#                 élaguer utilement l'arbre, sans repayer l'inférence à
#                 chaque nœud.
#   5. v1cld    : borne V1CLD par descente de sous-gradient (T_root/T_node iters).
#
# Chaque run imprime nodes/depth/fails/temps (cf. main_gecode.cpp). En plus du
# détail par run, ce script construit un TABLEAU RÉCAPITULATIF final (nœuds
# visités, gap relatif à la racine, temps total, temps BAB pur) en parsant la
# sortie de chaque run — les libellés/formats sont fixes côté C++
# (std::fixed/setprecision), cf. main_gecode.cpp:170-209.
#
# Soumettre :
#   sbatch compare_ref_v1ld_v1cld_gnn_gen.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_5methods_n20N100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

# ── Paramètres de l'instance générée (mêmes défauts que compare_ref_gnn_gen.sh
#    et compare_ref_gnn_song.sh : n=20/m=5/N=100, distribution Song/Pisinger) ─
# Surchageables via l'environnement (ex: N_SCEN=50 sbatch ...) pour tester
# d'autres tailles sans dupliquer le script.
N_ITEMS="${N_ITEMS:-20}"
M="${M:-5}"
N_SCEN="${N_SCEN:-100}"
EPSILON=0.10
ALPHA=0.5
R_PARAM=1000
CORRELATION=weakly_correlated
SEED=0

CKPT="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0197.pt"
T_ROOT=200
T_NODE=50
TIME_LIM_S=3300

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_v1ld_v1cld_gnn_gen — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Instance générée : n=$N_ITEMS  m=$M  N=$N_SCEN  eps=$EPSILON"
echo "    alpha=$ALPHA  R=$R_PARAM  correlation=$CORRELATION  seed=$SEED"
echo "  Checkpoint GNN : $CKPT  (backend=native, device=cuda)"
echo "  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s/run"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== Génération de l'instance (collect_s10-like) ==="
DATASET=$(python3 "$REPO/collect/gen_bench_instance.py" \
    --n "$N_ITEMS" --m "$M" --N "$N_SCEN" --seed "$SEED" \
    --alpha "$ALPHA" --R "$R_PARAM" --correlation "$CORRELATION")
echo "  Fichier : $DATASET"

TMP_REF=$(mktemp)
TMP_V1LD=$(mktemp)
TMP_GNN=$(mktemp)
TMP_GNN_ROOT=$(mktemp)
TMP_V1CLD=$(mktemp)
trap 'rm -f "$DATASET" "$TMP_REF" "$TMP_V1LD" "$TMP_GNN" "$TMP_GNN_ROOT" "$TMP_V1CLD"' EXIT

# ── Extraction des métriques d'un run à partir de sa sortie capturée ─────────
# Formats fixes côté C++ (main_gecode.cpp) :
#   "  Nœuds Gecode              : 13957 / 1048576 naïf  (1.33%)"
#   "  Borne UB à la racine      : 6733.3730  (gap vs optimal : 10.94%)"
#   "  Borne UB à la racine      : N/A (mode ref, pas de borne)"
#   "  Temps total               : 1.726 s"
#   "  Temps BAB pur             : 1.450 s  (12.0% du total)"
parse_metrics() {
    local file="$1"
    local nodes ttot tbab gap

    nodes=$(grep "Nœuds Gecode" "$file" | sed -E 's/.*: *([0-9]+) .*/\1/')
    ttot=$(grep "Temps total" "$file" | sed -E 's/.*: *([0-9.]+) s.*/\1/')
    tbab=$(grep "Temps BAB pur" "$file" | sed -E 's/.*: *([0-9.]+) s.*/\1/')

    if grep -q "gap vs optimal" "$file"; then
        gap=$(grep "Borne UB à la racine" "$file" | sed -E 's/.*gap vs optimal : ([0-9.]+)%.*/\1/')
        gap="${gap}%"
    else
        gap="N/A"
    fi

    printf '%s|%s|%s|%s\n' "$nodes" "$gap" "$ttot" "$tbab"
}

run_method() {
    local label="$1" tmp="$2"
    shift 2
    echo ""
    echo "=== [n=$N_ITEMS N=$N_SCEN] MODE=$label ==="
    "$@" | tee "$tmp"
}

run_method "ref" "$TMP_REF" \
    "$BINARY" "$DATASET" "$EPSILON" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

run_method "v1ld (sous-gradient, T_root=$T_ROOT, T_node=$T_NODE)" "$TMP_V1LD" \
    "$BINARY" "$DATASET" "$EPSILON" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1ld

run_method "gnn (backend=native, device=cuda)" "$TMP_GNN" \
    "$BINARY" "$DATASET" "$EPSILON" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native

run_method "gnn_root (backend=native, device=cuda, borne à la racine seulement)" "$TMP_GNN_ROOT" \
    "$BINARY" "$DATASET" "$EPSILON" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn_root non 0.1 "$CKPT" cuda python3 native

run_method "v1cld (sous-gradient, T_root=$T_ROOT, T_node=$T_NODE)" "$TMP_V1CLD" \
    "$BINARY" "$DATASET" "$EPSILON" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1cld

echo ""
echo "======================================================================"
echo "  TABLEAU RÉCAPITULATIF"
echo "======================================================================"
printf "%-8s | %14s | %12s | %16s | %16s\n" "Méthode" "Nœuds visités" "Gap racine" "Temps total (s)" "Temps BAB (s)"
printf -- "---------|----------------|--------------|------------------|------------------\n"
for entry in "ref:$TMP_REF" "v1ld:$TMP_V1LD" "gnn:$TMP_GNN" "gnn_root:$TMP_GNN_ROOT" "v1cld:$TMP_V1CLD"; do
    label="${entry%%:*}"
    file="${entry#*:}"
    IFS='|' read -r nodes gap ttot tbab <<< "$(parse_metrics "$file")"
    printf "%-8s | %14s | %12s | %16s | %16s\n" "$label" "$nodes" "$gap" "$ttot" "$tbab"
done
echo "======================================================================"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
