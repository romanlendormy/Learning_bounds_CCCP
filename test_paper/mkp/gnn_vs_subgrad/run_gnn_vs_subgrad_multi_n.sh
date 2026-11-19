#!/bin/bash
# test_paper/mkp/gnn_vs_subgrad/run_gnn_vs_subgrad_multi_n.sh — GNN vs sous-gradient à la racine,
# n ∈ {20, 25, 30} (n = 15 : run_gnn_vs_subgrad_n15.sh).
#
# Aller jusqu'à convergence du sous-gradient est trop coûteux à n ≥ 25 (≈ 0,5 s par itération
# à n = 25 avec ζ exact). On utilise donc l'arrêt anticipé GNN_COMPARE_BOUND :
#   (a) borne GNN en un seul forward à la racine (gnn_root / gnn_root_dld) ;
#   (b) sous-gradient à froid avec GNN_COMPARE_BOUND = <borne GNN> et T_root = 300 (plafond) :
#       il s'arrête dès que best_upper passe sous la borne GNN, t* = itérations exécutées.
# Sur n = 15, cette méthode redonne le t* obtenu par la trace complète.
#
# 180 configurations : formulation (sld, dld) × n ∈ {20, 25, 30} × N ∈ {100, 500, 1000} × graines 43..52.
# Indexation (TASK 0-indexé) : formulation = TASK < 90 ? sld : dld ; i = TASK % 90 ;
# n = [20, 25, 30][i / 30] ; N = [100, 500, 1000][(i % 30) / 10] ; graine = 43 + i % 10.
#
# Un seul job traite une plage de configurations (la QOS debug de Trillium n'accepte qu'un job
# soumis à la fois) : TASK_START..TASK_END (1-indexés, inclus) ou TASK_LIST="t1 t2 ...".
# Un lot de 15 configurations tient sous 2 h. Exemple :
#   sbatch --export=ALL,TASK_START=1,TASK_END=15 test_paper/mkp/gnn_vs_subgrad/run_gnn_vs_subgrad_multi_n.sh
# Pour n = 30 et N ≥ 500, augmenter les limites :
#   --export=ALL,TASK_LIST="...",GNN_TIME_LIMIT=120,SUBGRAD_TIME_LIMIT=300

#SBATCH --account=def-qcappart
#SBATCH --job-name=gnn_vs_subgrad_multin
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:50:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -uo pipefail   # pas de -e : une configuration en échec ne doit pas arrêter le lot
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"   # les binaires B&B doivent tourner depuis la racine du dépôt
source env/modules_trillium.sh
source env/setup_gpu_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

INSTANCES_DIR="${INSTANCES_DIR:-$REPO/test_paper/mkp/instances}"
OUT="${OUT_DIR:-${SCRATCH:-$REPO/runs}/test_paper_mkp/gnn_vs_subgrad_multin}"
mkdir -p "$OUT/logs"

BINARY="$REPO/build/solver_bab/bab_mkp_gecode"
N_ITEMS_VALUES=(20 25 30)
N_VALUES=(100 500 1000)
T_ROOT_MAX=300
# (a) limite du run GNN : doit couvrir le pré-chauffage TorchScript (≈ 10 s à n = 30, N = 100 ;
#     ≈ 60 s à n = 30, N = 500), qui précède le DFS.
GNN_TIME_LIMIT="${GNN_TIME_LIMIT:-20}"
# (b) limite interne du sous-gradient ; le timeout externe (+ 60 s) n'est qu'un filet :
#     le solveur doit sortir de lui-même pour imprimer sa borne racine.
TIME_LIMIT_S="${SUBGRAD_TIME_LIMIT:-90}"
TASK_START="${TASK_START:-1}"
TASK_END="${TASK_END:-180}"

run_task() {
    local TASK=$(($1 - 1)) FORMULATION I N_ITEMS N_SCEN SEED GNN_MODE CKPT
    local -a GNN_EXTRA_ARGS
    if [[ $TASK -lt 90 ]]; then FORMULATION="sld"; I=$TASK; else FORMULATION="dld"; I=$((TASK - 90)); fi
    N_ITEMS=${N_ITEMS_VALUES[$((I / 30))]}
    N_SCEN=${N_VALUES[$(((I % 30) / 10))]}
    SEED=$((43 + I % 10))
    if [[ "$FORMULATION" == "sld" ]]; then
        GNN_MODE="gnn_root"
        CKPT="models/mkp/sld/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
        GNN_EXTRA_ARGS=(cuda python3 native sd_edit)
    else
        GNN_MODE="gnn_root_dld"
        CKPT="models/mkp/dld/ckpt_dld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_dld_n10_20_500ep_ep0500.pt"
        GNN_EXTRA_ARGS=(cuda python3 native)
    fi
    local DATASET="$INSTANCES_DIR/instance_n${N_ITEMS}_m5_N${N_SCEN}_seed${SEED}.txt"
    [[ -f "$DATASET" ]] || { echo "[ERREUR] task=$1 instance introuvable : $DATASET" >&2; return 1; }

    local TAG="n${N_ITEMS}_N${N_SCEN}_seed${SEED}_${FORMULATION}"
    local GNN_LOG="$OUT/logs/gnn_root_${TAG}.log"
    echo "# task=$1 — $TAG — $(date)"

    # (a) borne GNN à la racine
    "$BINARY" "$DATASET" 0.10 30 "$GNN_TIME_LIMIT" 0 200 "$GNN_MODE" non 0.1 "$CKPT" "${GNN_EXTRA_ARGS[@]}" \
        > "$GNN_LOG" 2>&1
    local GNN_BOUND
    GNN_BOUND=$(grep "Borne UB à la racine" "$GNN_LOG" | grep -oP ': \K-?[0-9]+\.[0-9]+' | head -1)
    if [[ -z "$GNN_BOUND" ]] || awk "BEGIN{exit !($GNN_BOUND < 100)}"; then
        # Les bornes sont toujours >> 100 (échelle des profits) : une petite valeur trahit une
        # extraction erronée (le pourcentage de gap au lieu de la borne).
        echo "[ERREUR] task=$1 borne GNN absente ou suspecte : '$GNN_BOUND' ($GNN_LOG)" >&2
        return 1
    fi
    echo "  borne GNN : $GNN_BOUND"

    # (b) sous-gradient à froid avec arrêt anticipé. T_node = 1 : seule la racine compte,
    #     les nœuds suivants deviennent quasi gratuits.
    GNN_COMPARE_BOUND="$GNN_BOUND" ROOT_SUBGRAD_TRACE=1 timeout $((TIME_LIMIT_S + 60)) "$BINARY" \
        "$DATASET" 0.10 1 "$TIME_LIMIT_S" 0 "$T_ROOT_MAX" "$FORMULATION" \
        > "$OUT/logs/subgrad_${TAG}.stdout.log" 2> "$OUT/logs/subgrad_${TAG}.stderr.log"
    grep "racine" "$OUT/logs/subgrad_${TAG}.stdout.log" || echo "[ATTENTION] task=$1 pas de borne racine (sous-gradient)" >&2
    echo "  lignes SUBGRAD_TRACE : $(grep -c '^SUBGRAD_TRACE' "$OUT/logs/subgrad_${TAG}.stderr.log" || true)"
}

if [[ -n "${TASK_LIST:-}" ]]; then
    for T in $TASK_LIST; do run_task "$T"; done
else
    for T in $(seq "$TASK_START" "$TASK_END"); do run_task "$T"; done
fi
echo "# lot terminé — $(date)"
