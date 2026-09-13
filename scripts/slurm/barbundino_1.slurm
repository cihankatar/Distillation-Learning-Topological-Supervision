#!/bin/bash

#SBATCH -p barbun-cuda                                          # Kuyruk adi (GPU kuyrugu)
#SBATCH -A ckatar                                               # Kullanici adi
#SBATCH -J TopoSSL_S1                                           # Isin ismi (Seed 100)
#SBATCH -o logs/dinobarbun_s1_%A_%a.out                         # Cikti dosyasi (%A=Job ID, %a=Dizi Indeksi)
#SBATCH --gres=gpu:1                                            # 1 GPU
#SBATCH -N 1                                                    # 1 Node
#SBATCH -n 1                                                    # 1 Task
#SBATCH --cpus-per-task 20                                      # CPU cekirdek sayisi
#SBATCH --time=24:00:00                                         # Sure siniri (HH:MM:SS)

# 5 farkli etiket orani icin dizi (0: 5, 1: 10, 2: 20, 3: 104, 4: 208)
#SBATCH --array=0-4

module load miniconda3
conda activate segmentation
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs

# 5 farkli etiket orani
RATIOS=(0.0025 0.005 0.01 0.05 0.10)
LABELS=(5 10 20 104 208)

SEED=100
CURRENT_RATIO=${RATIOS[$SLURM_ARRAY_TASK_ID]}
CURRENT_LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================================="
echo "Job ID              : $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID)"
echo "Deney Serisi        : 1. Calistirma (Seed: $SEED)"
echo "Hedef Etiket Sayisi : $CURRENT_LABEL"
echo "Secilen Oran (-r)   : $CURRENT_RATIO"
echo "Calisan Node        : $SLURMD_NODENAME"
echo "Baslama Zamani      : $(date)"
echo "=========================================================="

export TOPODISTILL_SEED=$SEED
python train_ssl_pretrained.py -r $CURRENT_RATIO --seed $SEED
