# Persistence-Derived Pseudo-Masks for Label-Efficient Self-Distillation: TopoDistill for Dermoscopic Segmentation

Official implementation of the manuscript **“Persistence-Derived Pseudo-Masks for Label-Efficient Self-Distillation: TopoDistill for Dermoscopic Segmentation.”**

TopoDistill creates annotation-free lesion pseudo-masks from a cubical filtration and uses them as dense auxiliary supervision during global multi-crop DINO pretraining. The pretrained Att-Next encoder is transferred to skin-lesion segmentation under small labeled-data budgets.

## Pipeline

1. **Pseudo-mask extraction.** DullRazor hair suppression is followed by a robust lesion-evidence field combining color distance from an estimated skin reference, HSV saturation, and inverse brightness. A finite `H1` persistence class selects an image-specific threshold; active `H0` classes, spatial rejection, and protected morphology clean the mask.
2. **Self-distillation.** `train_dino.py` gives the teacher two global crops and the student the same two global crops plus four local crops. Global pooled features drive the DINO loss. A dense student head learns from the two geometrically aligned pseudo-masks using boundary-masked BCE plus Dice loss.
3. **Downstream segmentation.** `train_ssl_pretrained.py` freezes the pretrained encoder and fits the Att-Next decoder on a labeled subset. `train_random.py` is the same-backbone from-scratch baseline.

The optional ground-truth monitoring head is a detached diagnostic probe. It is disabled by default, does not update the encoder, and is not used for checkpoint selection. The encoder checkpoint is selected with validation DINO loss plus pseudo-mask loss.

## Repository structure

```text
augmentation/                  paired image/mask transformations
data/                          dataset classes and loaders
models/                        Att-Next encoder/decoder and heads
utils/                         losses, metrics, color fields, and mask methods
scripts/                       manuscript and W&B figure utilities
scripts/slurm/                 cluster templates for seeds 100, 200, and 300
cubical_complex_main.py        topology algorithm and interactive diagnostics
test_new.py                    batch pseudo-mask evaluation/generation
train_dino.py                  TopoDistill pretraining
train_ssl_pretrained.py        frozen-encoder downstream training
train_random.py                supervised from-scratch baseline
train_pseudo_supervised.py     direct training on pseudo-masks
test.py                        checkpoint evaluation and aggregation
config.yaml                    readable reference defaults
```

## Installation

Python 3.10 or newer is recommended. Install the correct PyTorch build for the target CUDA version, then install the repository requirements.

```bash
git clone https://github.com/cihankatar/Distillation-Learning-Topological-Supervision.git
cd Distillation-Learning-Topological-Supervision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local environment file and replace the example paths:

```bash
cp .env.example .env
set -a
source .env
set +a
```

`.env`, datasets, checkpoints, W&B files, and generated reports are ignored by Git.

## Data layout

`ML_DATA_ROOT` must point to the parent directory containing the datasets:

```text
ML_DATA_ROOT/
└── isic_2018_1/
    ├── train/
    │   ├── images/
    │   ├── masks/
    │   └── pmasks/
    ├── val/
    │   ├── images/
    │   ├── masks/
    │   └── pmasks/
    └── test/
        ├── images/
        ├── masks/
        └── pmasks/
```

Supported identifiers are `isic_2018_1`, `isic_2016_1`, `PH2Dataset`, `kvasir_1`, and `ham_1` (`HAM10000_1/` on disk). The pretraining loader aligns image, mask, and pseudo-mask filenames by a canonical sample identifier. Downstream image/mask directories must contain matching, consistently sortable filenames.

## Generate pseudo-masks

The canonical batch entry point is `test_new.py`. This command generates only the final `alpha=0.20` topology masks and writes them into each split’s `pmasks/` directory:

```bash
python test_new.py \
  --dataset isic2018 \
  --splits train val test \
  --topology-only \
  --save-topological-masks
```

Pseudo-masks are not written unless `--save-topological-masks` is supplied. To evaluate the five classical baselines on the same scalar field, omit `--topology-only`:

```bash
python test_new.py --dataset isic2018 --splits test
```

Use `--sample-size 20` for a smoke test. Each run appends to `<dataset>/test_new_results.txt` and also creates a timestamped report. Reports contain split-specific metrics, Betti errors, topologically correct rates, runtimes, method failures, and low-IoU image names.

`cubical_complex_main.py` remains an interactive per-image diagnostic program. Its four plot families are controlled independently with `SHOW_DIAGNOSTIC_PLOT`, `SHOW_COMPARISON_PLOT`, `SHOW_SCALAR_EXPLANATION_PLOT`, and `SHOW_CLEANUP_PLOT`.

## Pretraining

```bash
export TOPODISTILL_DATASET=isic_2018_1
export TOPODISTILL_SEED=932
python train_dino.py --epochs 300 --bsize 8 --lrate 0.0001
```

The default objective is:

```text
L_total = L_DINO + lambda_p(epoch) * (L_BCE + L_Dice)
```

`lambda_p` warms linearly to `1.0` during the first 20 epochs. Runtime settings include:

```bash
export TOPODISTILL_PMASK_SUBDIR=pmasks
export TOPODISTILL_PSEUDO_WEIGHT=1.0
export TOPODISTILL_PSEUDO_WARMUP_EPOCHS=20
export TOPODISTILL_BOUNDARY_IGNORE_RADIUS=2
export TOPODISTILL_WANDB_VIS_EPOCH_INTERVAL=25
export TOPODISTILL_ENABLE_GT_MONITOR=false
```

Scalar losses and schedules are logged every epoch. Image/crop/mask visualizations are logged every 25 epochs by default.

## Downstream training

Use the saved encoder explicitly when possible:

```bash
export TOPODISTILL_ENCODER_CHECKPOINT=/absolute/path/to/encoder_checkpoint.pth
python train_ssl_pretrained.py --epochs 503 --sratio 0.01 --seed 100
```

For ISIC2018, ratios `0.0025`, `0.005`, `0.01`, `0.05`, `0.10`, and `0.50` resolve to exactly 5, 10, 20, 104, 208, and 1040 labeled images. The from-scratch comparison is:

```bash
python train_random.py --epochs 450 --sratio 0.01 --seed 100
```

Direct training on pseudo-masks is available as a diagnostic comparison:

```bash
python train_pseudo_supervised.py \
  --dataset isic_2018_1 \
  --epochs 300 \
  --validation-target pseudo
```

The three `scripts/slurm/barbundino_*.slurm` files are cluster templates for seeds 100, 200, and 300. Submit them from the repository root after setting the environment variables; the scripts do not modify or reset the Git checkout.

## Evaluation

Evaluate one segmentation checkpoint:

```bash
python test.py \
  --checkpoint /absolute/path/to/checkpoint \
  --test-dataset isic_2018_1
```

Discover and aggregate several label-budget runs:

```bash
python test.py \
  --checkpoint-dir "$ML_DATA_OUTPUT/isic_1" \
  --ratios 0.0025 0.005 0.01 0.05 0.1 \
  --seeds 100 200 300 \
  --report-dir reports/downstream_test
```

The report directory includes per-checkpoint CSV data, aggregate CSV/JSON, Markdown, and a LaTeX table fragment. Add all actual experimental seeds to `--seeds`; do not describe three runs as five in a manuscript.

## Rebuild plots without retraining

Completed W&B histories can be exported directly into a journal-quality epoch plot with sample-standard-deviation bands:

```bash
python scripts/plot_wandb_curves.py \
  --metric validation/gt_monitor_iou \
  --group 'TopoDistill=entity/project/run1,entity/project/run2' \
  --group 'Self-Distillation=entity/project/run3,entity/project/run4' \
  --output figures/monitor_iou
```

This produces PNG, PDF, and CSV files without starting training. A valid `±1 SD` band requires multiple completed runs with the raw per-epoch metric. A dashboard screenshot or one averaged curve cannot reconstruct that uncertainty.

Additional figure utilities:

```bash
python scripts/generate_label_budget_qualitative.py --help
python scripts/train_dino_framework_visualization.py --help
```

## Weights & Biases

Use an existing `wandb login` session or provide `WANDB_API_KEY` only through the environment. For a local run:

```bash
export WANDB_MODE=offline
```

Never commit API keys. If a credential has appeared in Git history, revoke it in W&B and issue a new one; deleting it from the latest file does not erase the old commit.

## Reproducibility notes

- Training subsets are selected deterministically from `TOPODISTILL_SEED`/`--seed`.
- Pseudo-mask baselines in `test_new.py` receive the same robust scalar field.
- The final topology mask uses `alpha=0.20`; alternative alpha values must be selected on development data, not the held-out test set.
- Dataset files, pseudo-masks, and pretrained weights are not included.
- No open-source license is currently included; reuse and redistribution require the authors’ permission until a license is chosen.

## Citation

A BibTeX record will be added when publication metadata is available.
