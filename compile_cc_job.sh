#!/bin/bash
# compile_cc_job.sh — Job SLURM pour compiler la chaîne C++ (compile_cc.sh) sur rorqual.
#
# Soumettre (depuis la racine du dépôt, sur rorqual) :
#   sbatch compile_cc_job.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=compile_cc
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# WITH_CUDA=1 : lie aussi external/libtorch-cuda (backend GNN natif, device=cuda
# disponible en plus de cpu) — cf. gnn_v1ld_native.cpp. Le zip doit déjà être
# présent dans external/ (pas d'accès internet garanti sur un nœud de calcul) :
# téléchargé au préalable depuis le login node via
# WITH_CUDA=1 bash compile_cc.sh --download-only
export WITH_CUDA=1

bash compile_cc.sh
