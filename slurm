#!/bin/bash
#SBATCH -A csc662        
#SBATCH -J cone_beam_16cpu
#SBATCH --nodes=1                       
#SBATCH --gres=gpu:1                    
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -q debug                        
#SBATCH -o cone_beam_16cpu-%j.out
#SBATCH -e cone_beam_16cpu-%j.err

# ============================================================
# 1. LOAD MODULES (simplified for single GPU)
# ============================================================
module load PrgEnv-gnu
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
module unload darshan-runtime

# ============================================================
# 2. SET ENVIRONMENT VARIABLES (MOVE HERE!)
# ============================================================
export NUMBA_THREADING_LAYER='omp'
export MKL_NUM_THREADS=1
export OMP_PROC_BIND='spread'
export OMP_PLACES='threads'
export NUMBA_NUM_THREADS=16
export OMP_NUM_THREADS=16
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=hybrid
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================================
# 3. SETUP PYTHON ENVIRONMENT
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/csc662/world-shared/topcicekd/pytorch_env

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Numba version: $(python -c 'import numba; print(numba.__version__)')"

# ============================================================
# 4. PRE-WARM NUMBA
# ============================================================
echo "Setting up for $SLURM_CPUS_PER_TASK CPU cores"
echo "NUMBA_NUM_THREADS = $NUMBA_NUM_THREADS"
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"

echo "Pre-warming Numba compilation..."
python -c "
import numpy as np
from numba import jit
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def warmup_func(x):
    return x * 2
result = warmup_func(np.ones((10,10,10), dtype=np.float32))
print('Numba warmed up (result shape:', result.shape, ')')
"

# ============================================================
# 5. RUN YOUR SINGLE-GPU CODE
# ============================================================
echo "=========================================="
echo "Starting cone beam CT reconstruction"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo "Date: $(date)"
echo "=========================================="

time python main.py \
    --device=cuda \
    --iterations=100 \
    --lr=0.1 \
    --det_rows=128 \
    --det_channels=128 \
    --views=64 \
    --magnification=2.0 \
    --dsd_factor=3.0 \
    --block_size=2 2 2 \
    --phantom_scale=1.0 \
    --skip_adjoint_test \
    --output_dir=frontier_output_16cpu_${SLURM_JOB_ID}

echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="

# Show output files
echo -e "\nChecking output directory..."
ls -lh frontier_output_16cpu_${SLURM_JOB_ID}/ 2>/dev/null || echo "No output directory found"

# Quick check of log file
echo -e "\nLast 10 lines of log file:"
tail -10 frontier_output_16cpu_${SLURM_JOB_ID}/*.log 2>/dev/null || echo "No log file found yet"
