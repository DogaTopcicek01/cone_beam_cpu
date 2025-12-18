#!/bin/bash
#SBATCH -A csc662        
#SBATCH -J cone_beam_16cpu  # Changed job name
#SBATCH --nodes=1                       
#SBATCH --gres=gpu:1                    
#SBATCH --ntasks-per-node=1             
#SBATCH --cpus-per-task=16              # ← CHANGED to 16 CPUs
#SBATCH --mem=64G                       # ← ADDED: Less memory for test
#SBATCH -t 00:30:00                     # ← CHANGED: 30 min (shorter)
#SBATCH -q debug                        
#SBATCH -o cone_beam_16cpu-%j.out       # ← CHANGED output name  
#SBATCH -e cone_beam_16cpu-%j.err       # ← CHANGED error name

# ============================================================
# 1. LOAD MODULES (simplified for single GPU)
# ============================================================
module load PrgEnv-gnu
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

module unload darshan-runtime

# ============================================================
# 2. SETUP PYTHON ENVIRONMENT
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/csc662/world-shared/topcicekd/pytorch_env

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Numba version: $(python -c 'import numba; print(numba.__version__)')"

# ============================================================
# 3. CPU/GPU OPTIMIZATION FLAGS
# ============================================================
echo "Setting up for $SLURM_CPUS_PER_TASK CPU cores"

# Set threads for Numba parallel execution
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# GPU memory optimization
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=hybrid
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "NUMBA_NUM_THREADS = $NUMBA_NUM_THREADS"
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"

# ============================================================
# 4. RUN YOUR SINGLE-GPU CODE
# ============================================================

echo "=========================================="
echo "Starting cone beam CT reconstruction (16 CPU cores)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo "Date: $(date)"
echo "=========================================="

# Run your single-GPU code with command-line arguments
# Run your single-GPU code with ALL command-line arguments
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
echo "Output files created in frontier_output_16cpu_${SLURM_JOB_ID}/:"
ls -lh frontier_output_16cpu_${SLURM_JOB_ID}/ 2>/dev/null || echo "No output directory found"
