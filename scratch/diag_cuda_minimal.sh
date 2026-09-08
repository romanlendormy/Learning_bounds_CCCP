#!/bin/bash
# scratch/diag_cuda_minimal.sh — Diagnostic minimal : le GPU est-il visible
# indépendamment de notre pipeline (nvidia-smi, PyTorch, LibTorch C++ nu) ?
# Sert à trancher si "CUDA driver initialization failed" (vu dans
# sanity_check_cleanup.sh) est un problème d'infra ou lié a notre code/env.
#
# Usage : sbatch scratch/diag_cuda_minimal.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=diag_cuda
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=0:10:00
#SBATCH --output=scratch/logs/diag_cuda_%j.out
#SBATCH --error=scratch/logs/diag_cuda_%j.err

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"
mkdir -p scratch/logs

echo "======================================================================"
echo "  Nœud : $(hostname)   Job : ${SLURM_JOB_ID:-local}"
echo "  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<non défini>}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<non défini>}"
echo "======================================================================"

echo
echo "--- 1) nvidia-smi brut (avant tout module/env) ---"
nvidia-smi

echo
echo "--- 2) apres modules + setup_gpu_env.sh (meme env que le pipeline) ---"
module load StdEnv/2023 gcc/12.3 cmake/3.27.7 python/3.11 scipy-stack
source solver_bab/setup_gpu_env.sh
source "$HOME/envs/learning_bounds/bin/activate"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
nvidia-smi

echo
echo "--- 3) PyTorch voit-il CUDA (meme venv que le pipeline) ---"
python3 -c "
import torch
print('torch version         :', torch.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device name           :', torch.cuda.get_device_name(0))
    x = torch.randn(10, 10, device='cuda')
    print('tenseur cuda OK, sum  :', x.sum().item())
"

echo
echo "--- 4) LibTorch C++ nu voit-il CUDA (meme lib que les binaires bab_mkp_gecode) ---"
cat > /tmp/diag_cuda_cpp_$$.cpp << 'EOF'
#include <torch/torch.h>
#include <iostream>
int main() {
    std::cout << "cuda::is_available() = " << torch::cuda::is_available() << std::endl;
    std::cout << "cuda::device_count() = " << torch::cuda::device_count() << std::endl;
    if (torch::cuda::is_available()) {
        auto x = torch::randn({10, 10}, torch::device(torch::kCUDA));
        std::cout << "tenseur cuda OK, sum = " << x.sum().item<float>() << std::endl;
    }
    return 0;
}
EOF
g++ -std=c++17 -I"$REPO/external/libtorch-cuda/include" -I"$REPO/external/libtorch-cuda/include/torch/csrc/api/include" \
    -L"$REPO/external/libtorch-cuda/lib" -Wl,-rpath,"$REPO/external/libtorch-cuda/lib" \
    -o /tmp/diag_cuda_cpp_$$ /tmp/diag_cuda_cpp_$$.cpp -ltorch -ltorch_cpu -lc10 -ltorch_cuda -lc10_cuda \
    2>&1 | tail -30
if [[ -x /tmp/diag_cuda_cpp_$$ ]]; then
    /tmp/diag_cuda_cpp_$$
else
    echo "[ERREUR] compilation du test LibTorch C++ a échoué"
fi
rm -f /tmp/diag_cuda_cpp_$$ /tmp/diag_cuda_cpp_$$.cpp

echo
echo "--- 5) le binaire bab_mkp_gecode lui-meme, mode le plus simple possible (v1ld, pas de gnn) ---"
solver_bab/build/bab_mkp_gecode test_paper/instance_n15_m5_N100_seed43.txt 0.10 10 60 0 20 v1ld 2>&1 | tail -20
