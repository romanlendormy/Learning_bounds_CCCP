#!/bin/bash
# run_gen_ssp_test_paper_trillium_rorqual.sh — Génère le jeu de test SSP pour
# test_paper_trillium/ssp/ : grille E∈{4,6,8} × T∈{8,10,12} × N∈{50,100,500},
# 10 répétitions par combo (270 instances), calibrées par bissection sur le
# nombre de nœuds B&B (ssp_probe / Gecode base).
#
# Calibration validée le 2026-08-22 (voir test_paper_trillium/ssp/README.md) :
#   - epsilon = 0.1 pour toutes les instances
#   - fenêtre cible [nœuds B&B] :
#       E=4,T=10/12 -> [1e6,1e8] (E=4 sous [1e4,1e6] donne des instances
#         triviales, <0.2s, cf. docstring gnn_ssp/gen_ssp_instance_custom.py)
#       E=4,T=8     -> [1e4,1e6] (smoke test job 19418037 : [1e6,1e8]
#         INATTEIGNABLE pour ce combo précis -- oscille entre too_easy et
#         INFEASIBLE sans palier intermédiaire)
#       E=6,8 (tout T) -> [1e4,1e6] (fenêtre par défaut du repo)
#   - master_seed = 42, ep_id = 0..9 pour les 10 répétitions
#   - E=6,8 : rep0 réutilisé depuis gnn_ssp/test_data/ssp_native_test_E{E}T{T}N{N}_sweep/
#             (calibration d'origine non documentée, acceptée pour économiser du calcul)
#   - E=4   : les 10 répétitions sont régénérées from scratch (pas de réutilisation
#             du _sweep existant, qui utilise probablement l'ancienne fenêtre par
#             défaut et serait donc trivial, incohérent avec les 9 autres répétitions)
#
# Job array : un task par combo (E,T,N), 27 tasks (index 0-26), chaque task
# génère en série les répétitions manquantes de son combo.
#
# Soumettre (les 27 combos) :
#   sbatch run_gen_ssp_test_paper_trillium_rorqual.sh
#
# Smoke test (un seul combo, ex E=4,T=8,N=50 -> index 0) :
#   sbatch --array=0 run_gen_ssp_test_paper_trillium_rorqual.sh
#
# Relancer seulement les combos qui ont échoué (ex E=4,T=8 -> indices 0,1,2) :
#   sbatch --array=0,1,2 run_gen_ssp_test_paper_trillium_rorqual.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gen_ssp_test_paper_trillium
#SBATCH --array=0-26
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -uo pipefail   # pas de -e global : une répétition qui échoue ne doit pas tuer les autres

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
GEN_SCRIPT="$REPO/gnn_ssp/gen_ssp_instance_custom.py"
PROBE_BIN="$REPO/solver_bab/build_collect_cpu/ssp_probe"
OUT_ROOT="$REPO/test_paper_trillium/ssp"
VENV_PATH="$HOME/envs/learning_bounds"

EPSILON="0.1"
MASTER_SEED="42"
N_REPS=10   # ep_id 0..9

# ── Modules & venv ────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3.0 python/3.11 2>/dev/null || true
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$REPO/logs" "$OUT_ROOT"

if [ ! -x "$PROBE_BIN" ]; then
    echo "ERREUR : binaire introuvable : $PROBE_BIN"
    exit 1
fi

# ── Grille (E,T,N) depuis SLURM_ARRAY_TASK_ID (0-26) ──────────────────────────
EVALS=(4 6 8)
TVALS=(8 10 12)
NVALS=(50 100 500)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
E_IDX=$(( IDX / 9 ))
REM=$(( IDX % 9 ))
T_IDX=$(( REM / 3 ))
N_IDX=$(( REM % 3 ))

E="${EVALS[$E_IDX]}"
T="${TVALS[$T_IDX]}"
N="${NVALS[$N_IDX]}"

COMBO_DIR="$OUT_ROOT/E${E}_T${T}_N${N}"
mkdir -p "$COMBO_DIR"
PROV_FILE="$COMBO_DIR/PROVENANCE.txt"

echo "======================================================================"
echo "  gen_ssp_test_paper_trillium — $(date)"
echo "  Job ${SLURM_ARRAY_JOB_ID:-local}_${IDX}   Nœud : $(hostname)"
echo "  Combo : E=$E T=$T N=$N   -> $COMBO_DIR"
echo "======================================================================"

# ── Fenêtre de difficulté selon (E,T), et stratégie de réutilisation ─────────
# E=4,T=8 : smoke test (2026-08-22, job 19418037) a confirmé que [1e6,1e8] est
# INATTEIGNABLE pour ce combo précis (bissection oscille entre "too_easy"
# <=~24k nœuds et INFEASIBLE, aucun palier intermédiaire) -- on retombe sur la
# fenêtre par défaut [1e4,1e6], documentée comme fonctionnelle pour (E=4,T=8)
# dans l'étude de calibration de run_sweep_ssp_bound_efficacy_rorqual.sh.
# E=4,T=10/12 : fenêtre relevée [1e6,1e8], documentée fonctionnelle par le
# docstring de gen_ssp_instance_custom.py (évite la trivialité <0.2s).
if [ "$E" -eq 4 ] && [ "$T" -eq 8 ]; then
    TARGET_LO="1e4"
    TARGET_HI="1e6"
    START_REP=0
    REUSED=0
elif [ "$E" -eq 4 ]; then
    TARGET_LO="1e6"
    TARGET_HI="1e8"
    START_REP=0
    REUSED=0
else
    TARGET_LO="1e4"
    TARGET_HI="1e6"
    SWEEP_DIR="$REPO/gnn_ssp/test_data/ssp_native_test_E${E}T${T}N${N}_sweep"
    if [ -d "$SWEEP_DIR" ] && [ -f "$SWEEP_DIR/instance.txt" ]; then
        REP0_DIR="$COMBO_DIR/rep0"
        mkdir -p "$REP0_DIR"
        cp "$SWEEP_DIR/instance.txt" "$SWEEP_DIR/automaton.txt" "$SWEEP_DIR/cost.txt" "$SWEEP_DIR/demand.txt" "$REP0_DIR/"
        echo "[gen] rep0 réutilisé depuis $SWEEP_DIR"
        START_REP=1
        REUSED=1
    else
        echo "[gen] pas de _sweep existant pour E${E}T${T}N${N} -- génération complète 0..9"
        START_REP=0
        REUSED=0
    fi
fi

{
    echo "epsilon=$EPSILON master_seed=$MASTER_SEED target_lo=$TARGET_LO target_hi=$TARGET_HI"
    echo "generated=$(date -Iseconds)"
    if [ "$REUSED" -eq 1 ]; then
        echo "rep0=reused_from_sweep source=$SWEEP_DIR (calibration d'origine non documentée)"
    fi
} > "$PROV_FILE"

# ── Génération des répétitions manquantes ─────────────────────────────────────
# Repli : si ep_id=K échoue (bissection n'atteint pas la fenêtre en 15 essais
# d'automate), on retente avec ep_id=K+100, +200, ... (tirage différent, même
# E/T/N/epsilon/master_seed) pour combler les 10 répétitions sans changer les
# paramètres scientifiques. L'ep_id effectivement utilisé est tracé dans
# PROVENANCE.txt pour la reproductibilité.
FALLBACK_OFFSETS="0 100 200 300 400"

N_OK=0
N_FAIL=0
for (( K=START_REP; K<N_REPS; K++ )); do
    REP_DIR="$COMBO_DIR/rep${K}"
    REP_OK=0
    for OFF in $FALLBACK_OFFSETS; do
        TRY_EPID=$(( K + OFF ))
        echo "---- rep $K (ep_id=$TRY_EPID) -> $REP_DIR ----"
        if python "$GEN_SCRIPT" \
                --E "$E" --T "$T" --N "$N" \
                --epsilon "$EPSILON" \
                --ep_id "$TRY_EPID" \
                --master_seed "$MASTER_SEED" \
                --target_lo "$TARGET_LO" --target_hi "$TARGET_HI" \
                --probe_bin "$PROBE_BIN" \
                --out_dir "$REP_DIR" ; then
            echo "rep${K}=generated ep_id=${TRY_EPID}" >> "$PROV_FILE"
            N_OK=$(( N_OK + 1 ))
            REP_OK=1
            break
        else
            echo "[gen] ÉCHEC rep $K ep_id=$TRY_EPID (E=$E T=$T N=$N) -- essai de repli suivant"
            rm -rf "$REP_DIR"
        fi
    done
    if [ "$REP_OK" -eq 0 ]; then
        echo "rep${K}=FAILED (ep_id essayés: $(for o in $FALLBACK_OFFSETS; do echo -n "$((K+o)) "; done))" >> "$PROV_FILE"
        N_FAIL=$(( N_FAIL + 1 ))
    fi
done

echo "======================================================================"
echo "  Combo E=$E T=$T N=$N terminé — $(date)"
echo "  répétitions générées : $N_OK/$(( N_REPS - START_REP ))   échecs : $N_FAIL"
[ "$N_FAIL" -gt 0 ] && echo "  ATTENTION : $N_FAIL échec(s) -- fenêtre de difficulté probablement inatteignable pour ce combo"
echo "======================================================================"
