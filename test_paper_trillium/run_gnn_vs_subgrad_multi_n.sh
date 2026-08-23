#!/bin/bash
# run_gnn_vs_subgrad_multi_n.sh (Trillium) — extension de
# run_gnn_vs_subgrad_n15.sh à n∈{20,25,30} (n=15 déjà fait séparément).
#
# Utilise le mécanisme d'arrêt anticipé GNN_COMPARE_BOUND (cf.
# solve_v1ld/solve_v1cld target_bound, gecode_bab.hpp
# MKPSharedData::gnn_compare_bound) au lieu d'une trace complète jusqu'à
# T_root=3000 : à n=25, le coût par itération du sous-gradient V1LD (ζ exact
# B&B) est ~500x plus cher qu'à n=15 (cf. mémoire
# project_ccmkp_trillium_migration, probe_n25_v1ld job 822242 — ~0.5s/iter),
# donc courir jusqu'à convergence complète est impraticable. À la place :
#   (a) calcule d'abord la borne GNN root-only (comme avant).
#   (b) lance le sous-gradient avec GNN_COMPARE_BOUND=<borne GNN> et
#       T_root=300 (plafond) — s'arrête DÈS que best_upper franchit ce seuil,
#       t* = nombre d'itérations effectivement exécutées.
# Validé sur n=15/N=100/seed43 (job 822314) : reproduit exactement t*=10
# déjà connu par l'ancienne méthode (trace complète + post-traitement). Puis
# validé end-to-end sur n=20/N=100/seed43 (job 822320, ~1min38 pour les 2
# étapes) après correction d'un bug d'extraction (grep/sed capturait le
# pourcentage de gap au lieu de la borne absolue — job 822317).
#
# 180 configs au total : n∈{20,25,30} x N∈{100,500,1000} x seed∈43..52 x
# formulation∈{v1ld,v1cld}. Grille (TASK 0-indexé, 0..179) : formulation =
# TASK<90 ? v1ld : v1cld ; idx2 = TASK%90 ; n = [20,25,30][idx2/30] ;
# idx3 = idx2%30 ; N = [100,500,1000][idx3/10] ; seed = 43+idx3%10.
#
# QOS "debug" limite à 1 job SOUMIS à la fois par utilisateur
# (MaxSubmitPU=1) — un --array=1-180 est rejeté (chaque tâche d'array compte
# comme 1 job soumis). Solution : boucle interne sur une PLAGE de TASK dans
# un seul job (donc 1 seule soumission SLURM), répétée pour plusieurs
# tranches via une boucle de soumission séquentielle côté login node — cf.
# submit_multin_sequential.sh. Plage passée par TASK_START/TASK_END (env,
# 1-indexés inclusifs) ; défaut 1..180 si non fournis (NE PAS lancer tel
# quel sans découper : dépasserait la limite de temps debug de 2h).

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/def-qcappart/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_vs_subgrad_multin
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:50:00
# 01:50:00 (pas 00:20:00, valeur oubliee au 1er lancement — cause des 8/12
# TIMEOUT du 2026-08-22) : un lot de 15 taches peut prendre jusqu'a ~30min
# (pire cas n=30/N=1000 ~2min/tache), sous le plafond debug de 2h.
#SBATCH --output=/scratch/romanlen/gnn_vs_subgrad_multin/slurm_logs/slurm_%x_%j.out
#SBATCH --error=/scratch/romanlen/gnn_vs_subgrad_multin/slurm_logs/slurm_%x_%j.err

set -uo pipefail   # PAS -e : on veut inspecter/continuer même si un appel time-out

REPO="/project/def-qcappart/romanlen/Learning_bounds_CCCP"
cd "$REPO"

module load StdEnv/2023 gcc/12.3
source solver_bab/setup_gpu_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

SCRATCH_OUT="/scratch/romanlen/gnn_vs_subgrad_multin"
mkdir -p "$SCRATCH_OUT/logs" "$SCRATCH_OUT/slurm_logs"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
N_ITEMS_VALUES=(20 25 30)
N_VALUES=(100 500 1000)
T_ROOT_MAX=300
# GNN_TIME_LIMIT (env, defaut 20) : le pre-chauffage TorchScript (avant meme
# le debut du DFS) scale avec N ET n — 10.4s a n=30/N=100, mais 63s a
# n=30/N=500 (job 822936/823076 etc. : 15 taches n=30 x N in {500,1000} +
# 2 n=25 x N in {500,1000} sont restees a root_bound_value=0/echec repete
# avec time_limit=20, DEPASSE par le seul warmup, avant meme d'atteindre le
# DFS). Augmenter via --export=...,GNN_TIME_LIMIT=120 pour les configs
# lourdes plutot que de changer la valeur par defaut (qui gaspillerait du
# temps sur les 165 configs deja calculees avec time_limit=20, ou le warmup
# est largement sous cette valeur).
GNN_TIME_LIMIT="${GNN_TIME_LIMIT:-20}"
# TIME_LIMIT_S : borne interne du solveur — DOIT rester < le "timeout"
# externe (ligne GNN_COMPARE_BOUND=... ci-dessous) pour que le solveur ait
# la main pour imprimer proprement ses resultats (sortie normale du DFS via
# son propre check de time_limit) plutot que d'etre tue sec par SIGTERM
# (aucun handler cote C++ -> "Borne UB a la racine" jamais imprimee, meme
# si root_bound_value etait deja calcule). Bug reel rencontre le 2026-08-22
# (job 822324, n=30 : timeout 280 < TIME_LIMIT_S 300 => kill avant sortie
# propre, stdout vide).
# SUBGRAD_TIME_LIMIT (env, defaut 90) : cout/iteration scale aussi avec N
# (plus de scenarios par iteration ζ) — 90s tenait a N=100 mais insuffisant
# a N=500/1000 pour les configs n=30 (et 2 cas n=25). Augmenter via
# --export=...,SUBGRAD_TIME_LIMIT=300 pour ces configs lourdes.
TIME_LIMIT_S="${SUBGRAD_TIME_LIMIT:-90}"

TASK_START="${TASK_START:-1}"
TASK_END="${TASK_END:-180}"

run_task() {
    local TASK1=$1   # 1-indexé
    local TASK=$((TASK1 - 1))
    local FORMULATION IDX2 N_ITEMS IDX3 N_SCEN SEED
    if [[ $TASK -lt 90 ]]; then
        FORMULATION="v1ld"
        IDX2=$TASK
    else
        FORMULATION="v1cld"
        IDX2=$((TASK - 90))
    fi
    N_ITEMS=${N_ITEMS_VALUES[$((IDX2 / 30))]}
    IDX3=$((IDX2 % 30))
    N_SCEN=${N_VALUES[$((IDX3 / 10))]}
    SEED=$((43 + IDX3 % 10))

    local TESTDIR GNN_MODE SUBGRAD_MODE CKPT
    local -a GNN_EXTRA_ARGS
    if [[ "$FORMULATION" == "v1ld" ]]; then
        TESTDIR="$REPO/test_paper_trillium"
        GNN_MODE="gnn_root"
        SUBGRAD_MODE="v1ld"
        CKPT="$REPO/gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
        GNN_EXTRA_ARGS=(cuda python3 native sd_edit)
    else
        TESTDIR="$REPO/test_paper_trillium_v1cld"
        GNN_MODE="gnn_root_v1cld"
        SUBGRAD_MODE="v1cld"
        CKPT="$REPO/gnn_sd_edit/runs_v1cld/ckpt_v1cld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_v1cld_n10_20_500ep_ep0500.pt"
        GNN_EXTRA_ARGS=(cuda python3 native)
    fi

    local DATASET="$TESTDIR/instance_n${N_ITEMS}_m5_N${N_SCEN}_seed${SEED}.txt"
    if [[ ! -f "$DATASET" ]]; then
        echo "[ERREUR] task=$TASK1 instance introuvable : $DATASET" >&2
        return 1
    fi

    local TAG="n${N_ITEMS}_N${N_SCEN}_seed${SEED}_${FORMULATION}"
    local GNN_LOG="$SCRATCH_OUT/logs/gnn_root_${TAG}.log"
    local SUBGRAD_STDOUT="$SCRATCH_OUT/logs/subgrad_${TAG}.stdout.log"
    local SUBGRAD_STDERR="$SCRATCH_OUT/logs/subgrad_${TAG}.stderr.log"

    echo "======================================================================"
    echo "  gnn_vs_subgrad_multin — task=$TASK1 — $TAG — $(date)"
    echo "======================================================================"

    echo "--- (a) forward GNN root-only (mode=$GNN_MODE) ---"
    # time_limit=20 (5e argument) : la racine elle-meme est capturee au tout
    # premier appel more() (quasi instantane), MAIS le pre-chauffage
    # TorchScript (une passe JIT par taille n_g=1..n, AVANT le debut du DFS
    # et donc AVANT que time_limit ne soit jamais verifie) coute a lui seul
    # ~10.4s a n=30 (job 822332 avec time_limit=3 : timeout AVANT meme
    # d'atteindre le DFS, 0 noeud, root_bound_value=0 par defaut — bug reel
    # constate le 2026-08-22). 20s laisse une marge confortable au-dessus du
    # pre-chauffage tout en restant tres inferieur aux 30s d'origine, qui
    # laissaient en plus tourner inutilement l'exploration REF-apres-racine
    # (job 822324 : 479322 noeuds visites pour rien, cf. commit precedent).
    "$BINARY" "$DATASET" 0.10 30 "$GNN_TIME_LIMIT" 0 200 "$GNN_MODE" non 0.1 "$CKPT" "${GNN_EXTRA_ARGS[@]}" \
        > "$GNN_LOG" 2>&1

    local GNN_BOUND
    GNN_BOUND=$(grep "Borne UB à la racine" "$GNN_LOG" | grep -oP ': \K-?[0-9]+\.[0-9]+' | head -1)
    if [[ -z "$GNN_BOUND" ]]; then
        echo "[ERREUR] task=$TASK1 impossible d'extraire la borne GNN depuis $GNN_LOG" >&2
        return 1
    fi
    # Garde-fou : les bornes de ce problème sont toujours >> 100 (échelle des
    # profits) — si l'extraction a par erreur capturé le pourcentage de gap
    # au lieu de la valeur absolue (bug détecté et corrigé le 2026-08-22,
    # job 822317), la valeur sera visiblement trop petite.
    if awk "BEGIN{exit !($GNN_BOUND < 100)}"; then
        echo "[ERREUR] task=$TASK1 borne GNN suspecte (trop petite) : $GNN_BOUND" >&2
        return 1
    fi
    echo "  Borne GNN extraite : $GNN_BOUND"

    echo "--- (b) sous-gradient avec arrêt anticipé (mode=$SUBGRAD_MODE, GNN_COMPARE_BOUND=$GNN_BOUND, T_root=$T_ROOT_MAX) ---"
    # T_node=1 (3e argument, PAS 30) : seule la racine nous interesse — les
    # noeuds suivants (T_node iterations, PAS d'arret anticipe puisque
    # target_relative=1e18 hors racine) sont un pur gaspillage ici. A n=30,
    # chaque iteration coute ~1.7-5s (mesure job 822333 : un seul noeud a
    # profondeur 2, T_node=30, a pris 52s) — laisser T_node=30 fait exploser
    # le temps par noeud une fois sorti de la racine, pour un resultat qu'on
    # n'exploite meme pas. T_node=1 (min non-nul, evite l'overflow de
    # v1LD_star=+1e18 par defaut si la boucle ne s'executait jamais avec
    # T_node=0) rend chaque noeud quasi instantane.
    #
    # timeout externe = TIME_LIMIT_S + 60s de marge — cf. commentaire sur
    # TIME_LIMIT_S plus haut : le solveur doit pouvoir sortir seul via son
    # propre check de time_limit et imprimer ses resultats AVANT que ce
    # timeout dur (filet de securite, pas mecanisme normal) ne le tue.
    GNN_COMPARE_BOUND="$GNN_BOUND" ROOT_SUBGRAD_TRACE=1 timeout $((TIME_LIMIT_S + 60)) "$BINARY" \
        "$DATASET" 0.10 1 "$TIME_LIMIT_S" 0 "$T_ROOT_MAX" "$SUBGRAD_MODE" \
        > "$SUBGRAD_STDOUT" 2> "$SUBGRAD_STDERR"
    grep "racine" "$SUBGRAD_STDOUT" || echo "[ATTENTION] task=$TASK1 pas de ligne 'racine' dans $SUBGRAD_STDOUT" >&2
    local N_TRACE
    N_TRACE=$(grep -c "^SUBGRAD_TRACE" "$SUBGRAD_STDERR" || true)
    echo "  nb lignes SUBGRAD_TRACE : $N_TRACE"
    echo "Terminé — task=$TASK1 — $TAG — $(date)"
    echo ""
}

# TASK_LIST (env, optionnel) : liste explicite de TASK séparés par des
# espaces — prioritaire sur TASK_START/TASK_END. Sert à ne relancer QUE les
# configs manquantes/incomplètes d'un run précédent (cf. sanity check du
# 2026-08-22 : 8/12 lots TIMEOUT car --time=00:20:00 oublié, trop court pour
# un lot de 15 tâches — corrigé, mais on ne veut pas refaire les 180).
if [[ -n "${TASK_LIST:-}" ]]; then
    echo "Liste explicite traitée par ce job : $TASK_LIST"
    for T in $TASK_LIST; do
        run_task "$T"
    done
    echo "=== Job terminé (liste explicite) — $(date) ==="
else
    echo "Plage traitée par ce job : TASK $TASK_START..$TASK_END (sur 180)"
    for T in $(seq "$TASK_START" "$TASK_END"); do
        run_task "$T"
    done
    echo "=== Job terminé (TASK $TASK_START..$TASK_END) — $(date) ==="
fi
