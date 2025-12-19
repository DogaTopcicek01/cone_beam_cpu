
#!/bin/bash
#SBATCH -A csc662
#SBATCH -J cone_beam_1gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -o cone_beam-%j.out
#SBATCH -e cone_beam-%j.err

# ============================================================
# 1. LOAD MODULES
# ============================================================
module load PrgEnv-gnu
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
module unload darshan-runtime

# ============================================================
# 2. ENVIRONMENT VARIABLES
# ============================================================
export NUMBA_THREADING_LAYER=omp
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=hybrid

# ============================================================
# 3. PYTHON ENV
# ============================================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/csc662/world-shared/topcicekd/pytorch_env

echo "Python: $(which python)"
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("HIP available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
EOF

# ============================================================
# 4. RUN
# ============================================================
echo "Starting job $SLURM_JOB_ID on $(hostname)"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

srun python main.py \
    --iterations 100 \
    --lr 0.1 \
    --det_rows 128 \
    --det_channels 128 \
    --views 64 \
    --magnification 2.0 \
    --dsd_factor 3.0 \
    --block_size 1 1 1 \
    --phantom_scale 1.0 \
    --skip_adjoint_test \
    --output_dir frontier_output_${SLURM_JOB_ID}
