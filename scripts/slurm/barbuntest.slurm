#!/bin/bash

#SBATCH -p barbun-cuda
#SBATCH -A ckatar
#SBATCH -J TopoTest
#SBATCH -o barbuntest_%j.out
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH --time=08:00:00

set -euo pipefail

module load miniconda3
conda activate segmentation

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CHECKPOINT_DIR="${ML_DATA_OUTPUT:?ML_DATA_OUTPUT must be set}/isic_1"
REPORT_DIR="${CHECKPOINT_DIR}/reports/downstream_test_${SLURM_JOB_ID}"

if [[ -z "${ML_DATA_ROOT:-}" ]]; then
    echo "ERROR: ML_DATA_ROOT must point to the directory containing isic_2018_1."
    exit 2
fi

cd "${REPO_DIR}"
mkdir -p "${REPORT_DIR}"

echo "Job ID          : ${SLURM_JOB_ID}"
echo "Node            : ${SLURMD_NODENAME}"
echo "Checkpoint dir  : ${CHECKPOINT_DIR}"
echo "Data root       : ${ML_DATA_ROOT}"
echo "Report dir      : ${REPORT_DIR}"
echo "Started         : $(date --iso-8601=seconds)"

# One job evaluates all 15 models so mean and sample standard deviation can be
# calculated across seeds in the same report.
python -u test.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --dataset isic_2018_1 \
    --test-dataset isic_2018_1 \
    --data-root "${ML_DATA_ROOT}" \
    --ratios 0.0025 0.005 0.01 0.05 0.1 \
    --seeds 100 200 300 \
    --batch-size 8 \
    --workers 8 \
    --image-size 256 \
    --threshold 0.5 \
    --metric-reduction legacy-batch \
    --device cuda \
    --report-dir "${REPORT_DIR}"

echo "Completed       : $(date --iso-8601=seconds)"
echo "Paper table     : ${REPORT_DIR}/table3_topodistill.tex"
echo "Readable report : ${REPORT_DIR}/report.md"
