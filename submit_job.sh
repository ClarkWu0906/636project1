#!/bin/bash
#==============================================================================
# CSCE 636 Project 1 — SLURM Job Script for TAMU Grace HPC
#==============================================================================
# Submit:   sbatch submit_job.sh
# Monitor:  squeue -u $USER
# Output:   slurm-<jobid>.out  (stdout+stderr)
#==============================================================================

#SBATCH --job-name=csce636_moe
#SBATCH --output=csce636_%j.out
#SBATCH --error=csce636_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# ---- Email notifications (optional, uncomment + set your email) ----
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your_email@tamu.edu

echo "============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "CPUs/task:     $SLURM_CPUS_PER_TASK"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "============================================="

# ---- Load modules ----
# Grace uses module system for software. Adjust versions as available.
module purge
module load GCC/12.3.0
module load CUDA/12.1.1
module load Python/3.11.3
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# If modules above don't exist on your Grace setup, try:
#   module spider Python
#   module spider PyTorch
#   module spider CUDA
# and adjust accordingly. Alternatively, use a conda environment:
#
#   module load Anaconda3
#   conda activate my_torch_env

# ---- Verify GPU ----
echo ""
echo "Python: $(which python3)"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ---- Install any missing packages ----
pip install --user scikit-learn scipy numpy 2>/dev/null

# ---- Set paths ----
DATA_DIR="$SLURM_SUBMIT_DIR"     # directory where data files live
OUTPUT_DIR="$SLURM_SUBMIT_DIR/output_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Data dir:   $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# ---- Run training ----
python3 "$SLURM_SUBMIT_DIR/final_model.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    --base_epochs 1000 \
    --weak_epochs 1200 \
    --ensemble_epochs 1200 \
    --base_runs 15 \
    --hard_runs 15 \
    --retrain_runs 15 \
    --ensemble_size 15 \
    --lp_per_group 10000 \
    --lp_weak 15000 \
    --batch_size 512 \
    --cost_threshold 0.0 \
    --num_workers 4

echo ""
echo "============================================="
echo "Job finished: $(date)"
echo "Output dir:   $OUTPUT_DIR"
echo "============================================="

# ---- Copy key outputs back to submit dir for convenience ----
cp "$OUTPUT_DIR/moe_model.pth" "$SLURM_SUBMIT_DIR/moe_model_grace.pth" 2>/dev/null
cp "$OUTPUT_DIR/predicted_mHeights" "$SLURM_SUBMIT_DIR/predicted_mHeights_grace" 2>/dev/null
echo "Copied moe_model_grace.pth and predicted_mHeights_grace to submit dir."
