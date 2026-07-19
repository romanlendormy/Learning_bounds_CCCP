#!/bin/bash
# debug_test_tf32_scale_fixes_n20_N20.sh — teste 2 hypothèses pour le bug de
# faux-optimal du B&B GNN batché sur n=20/N=20 (5130.397 déclaré OPTIMAL au
# lieu du vrai 5166.381, cf. mémoire
# project_gnn_batched_false_optimal_bug_investigation) :
#
#   TEST 1 — TF32 cuBLAS/cuDNN désactivé (déjà codé dans
#            gnn_sd_edit_native_batched.cpp, marqueur TEST_TF32_DISABLE).
#            SCALE reste à 1000 (valeur de production).
#   TEST 2 — SCALE 1000 -> 10000 (couvre les 4 décimales réelles du
#            dataset), TF32 remis à sa valeur par défaut (test 1 reverté
#            avant celui-ci pour isoler les deux hypothèses).
#
# Le script est AUTO-RÉVERSIBLE : à la fin, gecode_bab.hpp et
# gnn_sd_edit_native_batched.cpp sont remis dans leur état de départ
# (SCALE=1000, pas de désactivation TF32), et les binaires sont reconstruits
# une dernière fois dans cet état propre — peu importe le résultat des tests.
# C'est Claude qui décide ENSUITE, à la main, s'il faut réappliquer l'un des
# deux fixs de façon permanente, selon ce que ce script rapporte.
#
# Soumettre :
#   sbatch debug_test_tf32_scale_fixes_n20_N20.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=debug_test_tf32_scale_fixes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -uo pipefail   # pas de -e : un test qui échoue ne doit pas empêcher le revert final

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S20.txt"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
BATCHED="$REPO/solver_bab/build/bab_mkp_gecode_gnn_batched"
PROD="$REPO/solver_bab/build/bab_mkp_gecode"
TARGET_OPT="5166.381"

HPP="solver_bab/mkp/gecode_bab.hpp"
BATCHED_CPP="solver_bab/mkp/gnn_sd_edit_native_batched.cpp"

echo "======================================================================"
echo "  debug_test_tf32_scale_fixes_n20_N20 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

check_result() {
  local label="$1" out="$2"
  local val
  val=$(grep "Valeur optimale" "$out" | grep -oE '[0-9]+\.[0-9]+' | head -1)
  echo "  -> valeur obtenue : ${val:-<aucune>}"
  if [ -z "$val" ]; then
    echo "  -> [$label] ÉCHEC (pas de résultat parsable)"
    return 1
  fi
  # comparaison à 0.01 près
  python3 -c "import sys; v=float('$val'); sys.exit(0 if abs(v-$TARGET_OPT) < 0.01 else 1)"
  if [ $? -eq 0 ]; then
    echo "  -> [$label] CORRIGÉ : $val ≈ $TARGET_OPT"
    return 0
  else
    echo "  -> [$label] TOUJOURS FAUX : $val != $TARGET_OPT"
    return 1
  fi
}

echo ""
echo "######################################################################"
echo "  TEST 1 — TF32 cuBLAS/cuDNN désactivé (SCALE=1000, inchangé)"
echo "######################################################################"
grep -n "TEST_TF32_DISABLE" "$BATCHED_CPP" || echo "[ATTENTION] marqueur TEST_TF32_DISABLE absent, le code n'a pas l'air modifié"

cmake --build solver_bab/build --target bab_mkp_gecode_gnn_batched -j4 2>&1 | tail -30
if [ ! -x "$BATCHED" ]; then echo "[ERREUR] build échoué, binaire absent"; fi

OUT1=logs/test1_tf32_disabled_${SLURM_JOB_ID:-local}.out
"$BATCHED" "$DATASET" "$CKPT" 0.10 300 0 cuda 1 > "$OUT1" 2>&1
cat "$OUT1"
TEST1_PASS=1
check_result "TEST1 TF32-disabled" "$OUT1" && TEST1_PASS=0

echo ""
echo "######################################################################"
echo "  Revert TEST 1 (TF32 remis par défaut) avant TEST 2"
echo "######################################################################"
sed -i '/TEST_TF32_DISABLE/,/setAllowTF32CuDNN(false);/d' "$BATCHED_CPP"
grep -n "TF32" "$BATCHED_CPP" || echo "  (aucune ligne TF32 restante — revert OK)"

echo ""
echo "######################################################################"
echo "  TEST 2 — SCALE 1000 -> 10000 (TF32 par défaut)"
echo "######################################################################"
grep -n "static constexpr int SCALE" "$HPP"
sed -i 's/static constexpr int SCALE = 1000;/static constexpr int SCALE = 10000;/' "$HPP"
grep -n "static constexpr int SCALE" "$HPP"

cmake --build solver_bab/build --target bab_mkp_gecode_gnn_batched -j4 2>&1 | tail -30
cmake --build solver_bab/build --target bab_mkp_gecode -j4 2>&1 | tail -30

OUT2=logs/test2_scale10000_batched_${SLURM_JOB_ID:-local}.out
"$BATCHED" "$DATASET" "$CKPT" 0.10 300 0 cuda 1 > "$OUT2" 2>&1
cat "$OUT2"
TEST2_PASS=1
check_result "TEST2 SCALE=10000 (batché)" "$OUT2" && TEST2_PASS=0

echo ""
echo "--- sanity : mono (production) avec SCALE=10000, doit rester correct ---"
OUT2B=logs/test2_scale10000_mono_${SLURM_JOB_ID:-local}.out
"$PROD" "$DATASET" 0.10 50 60 0 200 gnn non 0.1 "$CKPT" cuda python3 native sd_edit > "$OUT2B" 2>&1
cat "$OUT2B"
check_result "TEST2 SCALE=10000 (mono, sanity)" "$OUT2B"

echo ""
echo "######################################################################"
echo "  Revert TEST 2 (SCALE remis à 1000) — état propre final"
echo "######################################################################"
sed -i 's/static constexpr int SCALE = 10000;/static constexpr int SCALE = 1000;/' "$HPP"
grep -n "static constexpr int SCALE" "$HPP"

echo ""
echo "--- rebuild final dans l'état d'origine (SCALE=1000, pas de TF32-disable) ---"
cmake --build solver_bab/build --target bab_mkp_gecode_gnn_batched -j4 2>&1 | tail -20
cmake --build solver_bab/build --target bab_mkp_gecode -j4 2>&1 | tail -20

echo ""
echo "======================================================================"
echo "  RÉSUMÉ"
echo "======================================================================"
echo "  TEST1 (TF32 désactivé, SCALE=1000)  : $([ $TEST1_PASS -eq 0 ] && echo CORRIGÉ || echo 'toujours faux')"
echo "  TEST2 (SCALE=10000, TF32 défaut)     : $([ $TEST2_PASS -eq 0 ] && echo CORRIGÉ || echo 'toujours faux')"
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
