# setup_gpu_env.sh — source (do not execute) before running a B&B binary built with
# -DGNN_DEVICE=cuda against external/libtorch-cuda (LibTorch 2.13.0 + CUDA 12.6).
#
# Why: unlike older releases, the LibTorch >= 2.13 "shared-with-deps" archive no longer bundles
# cuBLAS / cuDNN / cuSPARSE / CUPTI / NCCL / NVSHMEM; they come from the cluster modules.
# They are TRANSITIVE dependencies of libtorch_cuda.so, so the RUNPATH baked into our binaries
# does not cover them: LD_LIBRARY_PATH is required at run time.
# libnvshmem_host.so.3 is not available on the cluster; a stub built with the B&B
# (solver_bab/nvshmem_stub/, output in $BUILD_DIR/solver_bab) is used instead —
# it is never called for single-GPU inference.
#
# Usage:
#   source env/setup_gpu_env.sh
#   ./build/solver_bab/<binary> ... cuda

module load cuda/12.6 cudnn/9.10.0.56 cusparselt/0.7.1.0

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
BUILD_DIR="${BUILD_DIR:-$REPO/build}"

export LD_LIBRARY_PATH="$BUILD_DIR/solver_bab:$EBROOTCUDA/targets/x86_64-linux/lib:$EBROOTCUDA/extras/CUPTI/lib64:$EBROOTCUSPARSELT/lib:$EBROOTCUDNN/lib:$EBROOTNCCL/lib:$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
