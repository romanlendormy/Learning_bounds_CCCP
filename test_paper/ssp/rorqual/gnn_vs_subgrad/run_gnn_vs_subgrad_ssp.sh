#!/bin/bash
# run_gnn_vs_subgrad_ssp.sh — Portage CCMKP -> CCSSP de l'expérience "GNN
# forward vs sous-gradient (t*)" (cf. test_paper/mkp/trillium/gnn_vs_subgrad/run_gnn_vs_subgrad_multi_n.sh,
# parse_gnn_vs_subgrad.py côté MKP, mécanisme GNN_ROOT/gnn_compare_bound porté
# dans solver_bab/ssp/gecode_bab_ssp.hpp/.cpp + test_gnn_dld_bab.cpp).
#
# Portée demandée : DLD uniquement, (E,T) in {(4,8),(6,10),(8,12)},
# N in {100,500}, rep 0..9 -- 60 instances (test_paper/ssp/rorqual/ssp/).
#
# Pour chaque instance :
#   (a) calcule la borne GNN root-only (mode=gnn_root, device=cpu -- pas
#       besoin de GPU pour UN SEUL forward, et évite toute contention avec la
#       campagne trillium_ssp_ref/trillium_ssp_gnn_batched en cours sur les
#       partitions GPU au moment où ce script a été écrit, 2026-08-23).
#   (b) lance DLD (mode=dld) avec GNN_COMPARE_BOUND=<borne de (a)> et
#       ROOT_SUBGRAD_TRACE=1, T_root=300 T_node=1 -- s'arrête dès que le
#       sous-gradient dépasse la borne GNN, t* = nb d'itérations.
#
# Binaire : build ISOLÉ (solver_bab/build_ssp_gnnroot_sanity +
# solver_generic/build_cc_ssp_gnnroot), PAS solver_bab/build_gpu_ssp ni
# solver_generic/build_cc -- ces derniers sont activement reconstruits à
# chaque tâche par run_ref_array_ssp.sh/run_gnn_batched_array_ssp.sh (jobs
# 19428717/19428718 en cours au moment de l'écriture) ; les toucher aurait
# risqué une course d'écriture concurrente sur le même répertoire de build
# (cf. mémoire project_solver_generic_stale_build_trap -- même piège, version
# "deux processus qui rebuild en même temps", pas juste "code jamais rebuild").
# Construit une fois, réutilisé pour les 60 instances (pas de rebuild par
# tâche, contrairement aux scripts ref/gnn_batched).
#
# Calibration (validée manuellement le 2026-08-23 avant de lancer le sweep,
# cf. discussion -- sanity check sur E4_T8_N50 ET E8_T12_N500, la plus grosse
# config du sweep) :
#   - gnn_root : 1 seul forward, ~3s sur E8_T12_N500 (device=cpu) -- time_lim=60
#     laisse une marge large.
#   - dld root (T_root=300, T_node=1) : converge/plafonne en ~61 itérations
#     sur E8_T12_N500 (la plus grosse config), bien avant T_root=300 -- pas de
#     signe que le budget doive être augmenté. time_lim=90 laisse une marge
#     large (le reste de l'arbre, T_node=1, peut ensuite tourner jusqu'au
#     time_lim sans conséquence -- seule la racine nous intéresse, cf. note
#     ci-dessous sur le statut TIMEOUT attendu et sans importance).
#   - IMPORTANT (déjà observé au sanity check) : t* est FRÉQUEMMENT égal à 1 --
#     vérifié que ce n'est PAS un bug (mécanisme testé avec plusieurs seuils
#     artificiels, répond correctement -- cf. discussion), mais un résultat
#     probablement réel : le checkpoint GNN SSP (gnn_ssp/runs/train_scaled/
#     ckpt_scaled_dld_..._ep0150.pt) n'est entraîné QUE sur E in {3,4},
#     T in {8,12}, N in {20,50} -- (E6,T10)/(E8,T12) et N in {100,500} sont
#     largement hors distribution pour ce checkpoint. Rapporter honnêtement le
#     taux de censure/t*=1 plutôt que s'attendre à un résultat façon MKP.
#
# QOS/soumission : job sbatch UNIQUE (pas de --array) avec boucle interne sur
# les 60 configs -- évite toute question de limite de soumissions simultanées
# (cf. protocole -- QOS debug MaxSubmitPU=1, prudence par défaut même si ce
# job n'utilise pas cette QOS).

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_vs_subgrad_ssp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=test_paper/ssp/rorqual/gnn_vs_subgrad/logs/slurm_%x_%j.out
#SBATCH --error=test_paper/ssp/rorqual/gnn_vs_subgrad/logs/slurm_%x_%j.err

set -uo pipefail   # PAS -e : on veut inspecter/continuer même si un appel time-out

REPO="/project/6062881/romanlen/Learning_bounds_CCCP"
cd "$REPO"

module load StdEnv/2023 gcc/12.3 cuda/12.6 cudnn/9.10.0.56 cusparselt/0.7.1.0
source solver_bab/setup_gpu_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

LOGS_DIR="$REPO/test_paper/ssp/rorqual/gnn_vs_subgrad/logs"
mkdir -p "$LOGS_DIR"

BINARY="$REPO/solver_bab/build_ssp_gnnroot_sanity/test_gnn_dld_bab"
CKPT="$REPO/gnn_ssp/runs/train_scaled/ckpt_scaled_dld_E3_T12_N20_E3_T12_N50_E4_T12_N20_E4_T12_N50_E4_T8_N20_E4_T8_N50_lr0.0003_bs32_h64_seed42_v6_periodpool_1k_ep0150.pt"

SIZES=("4 8" "6 10" "8 12")
N_VALUES=(100 500)
REPS=(0 1 2 3 4 5 6 7 8 9)

GNN_TIME_LIMIT="${GNN_TIME_LIMIT:-60}"
SUBGRAD_TIME_LIMIT="${SUBGRAD_TIME_LIMIT:-90}"
T_ROOT_MAX="${T_ROOT_MAX:-300}"

TASK_START="${TASK_START:-1}"
TASK_END="${TASK_END:-60}"

run_task() {
    local TASK1=$1   # 1-indexé
    local TASK=$((TASK1 - 1))
    # TASK (0-indexé, 0..59) : idxN = TASK/30 (0 ou 1 -> N), idx2 = TASK%30,
    # idxE = idx2/10 (0,1,2 -> taille), rep = idx2%10.
    local IDX_N=$((TASK / 30))
    local IDX2=$((TASK % 30))
    local IDX_E=$((IDX2 / 10))
    local REP=$((IDX2 % 10))

    local N_SCEN=${N_VALUES[$IDX_N]}
    local ET_PAIR=(${SIZES[$IDX_E]})
    local E_VAL=${ET_PAIR[0]}
    local T_VAL=${ET_PAIR[1]}

    local TAG="E${E_VAL}_T${T_VAL}_N${N_SCEN}_rep${REP}"
    local DATASET="$REPO/test_paper/ssp/rorqual/ssp/E${E_VAL}_T${T_VAL}_N${N_SCEN}/rep${REP}"

    if [[ ! -d "$DATASET" ]]; then
        echo "[ERREUR] task=$TASK1 instance introuvable : $DATASET" >&2
        return 1
    fi

    local GNN_LOG="$LOGS_DIR/gnn_vs_subgrad_root_${TAG}.log"
    local SUBGRAD_STDOUT="$LOGS_DIR/gnn_vs_subgrad_dld_${TAG}.stdout.log"
    local SUBGRAD_STDERR="$LOGS_DIR/gnn_vs_subgrad_dld_${TAG}.stderr.log"

    echo "======================================================================"
    echo "  gnn_vs_subgrad_ssp — task=$TASK1 — $TAG — $(date)"
    echo "======================================================================"

    echo "--- (a) forward GNN root-only (mode=gnn_root, device=cpu) ---"
    timeout $((GNN_TIME_LIMIT + 60)) "$BINARY" "$DATASET" "$CKPT" 0.0 "$GNN_TIME_LIMIT" gnn_root cpu 200 30 \
        > "$GNN_LOG" 2>&1

    local GNN_BOUND
    GNN_BOUND=$(grep -oP 'root_bound_value=\K-?[0-9.]+' "$GNN_LOG" | head -1)
    if [[ -z "$GNN_BOUND" ]]; then
        echo "[ERREUR] task=$TASK1 impossible d'extraire la borne GNN depuis $GNN_LOG" >&2
        return 1
    fi
    # Garde-fou : les coûts salariaux de ce problème sont toujours > 1 (jamais
    # un pourcentage de gap, cf. piège grep/sed déjà rencontré côté MKP le
    # 2026-08-22 -- ici root_bound_value=%.6f est une clé=valeur non ambiguë,
    # mais on garde le garde-fou par prudence).
    if awk "BEGIN{exit !($GNN_BOUND < 0.5)}"; then
        echo "[ERREUR] task=$TASK1 borne GNN suspecte (trop petite) : $GNN_BOUND" >&2
        return 1
    fi
    echo "  Borne GNN extraite : $GNN_BOUND"

    echo "--- (b) sous-gradient DLD avec arrêt anticipé (GNN_COMPARE_BOUND=$GNN_BOUND, T_root=$T_ROOT_MAX, T_node=1) ---"
    GNN_COMPARE_BOUND="$GNN_BOUND" ROOT_SUBGRAD_TRACE=1 timeout $((SUBGRAD_TIME_LIMIT + 60)) "$BINARY" \
        "$DATASET" "$CKPT" 0.0 "$SUBGRAD_TIME_LIMIT" dld cpu "$T_ROOT_MAX" 1 \
        > "$SUBGRAD_STDOUT" 2> "$SUBGRAD_STDERR"
    grep "root_bound_value" "$SUBGRAD_STDOUT" || echo "[ATTENTION] task=$TASK1 pas de root_bound_value dans $SUBGRAD_STDOUT" >&2
    local N_TRACE
    N_TRACE=$(grep -c "^SUBGRAD_TRACE" "$SUBGRAD_STDERR" || true)
    echo "  nb lignes SUBGRAD_TRACE : $N_TRACE"
    echo "Terminé — task=$TASK1 — $TAG — $(date)"
    echo ""
}

echo "Plage traitée par ce job : TASK $TASK_START..$TASK_END (sur 60)"
for T in $(seq "$TASK_START" "$TASK_END"); do
    run_task "$T"
done
echo "=== Job terminé (TASK $TASK_START..$TASK_END) — $(date) ==="
