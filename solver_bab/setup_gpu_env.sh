# setup_gpu_env.sh — À sourcer (pas exécuter) avant de lancer un binaire
# bab_mkp_gecode/gnn_native_probe compilé avec -DGNN_DEVICE=cuda contre
# external/libtorch-cuda (2.13.0+cu126).
#
# Pourquoi : contrairement à libtorch 2.5.1 (vendored, self-contained),
# le zip "shared-with-deps" de libtorch >= 2.13 ne bundle plus cublas/
# cudnn/cusparse/cupti/nccl/nvshmem dans son propre lib/ — il faut les
# fournir via les modules système du cluster. Le RPATH/RUNPATH baké au
# link ne suffit PAS : ce sont des dépendances TRANSITIVES de
# libtorch_cuda.so (pas des besoins directs de nos binaires), et
# DT_RUNPATH (contrairement à l'ancien DT_RPATH) ne s'applique qu'aux
# besoins DIRECTS de l'objet qui le porte — LD_LIBRARY_PATH est donc
# obligatoire à l'exécution. Cf. mémoire "libtorch213-cuda-validated".
#
# Usage :
#   source solver_bab/setup_gpu_env.sh
#   ./solver_bab/build/bab_mkp_gecode ... gnn <ckpt> cuda native

module load cuda/12.6 cudnn/9.10.0.56 cusparselt/0.7.1.0

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

export LD_LIBRARY_PATH="$REPO/external/nvshmem_stub/lib:$EBROOTCUDA/targets/x86_64-linux/lib:$EBROOTCUDA/extras/CUPTI/lib64:$EBROOTCUSPARSELT/lib:$EBROOTCUDNN/lib:$EBROOTNCCL/lib:$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
