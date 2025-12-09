# POC_Dataset Classification Pipeline with ResNet

> All code was written by me, and only some phrases in the README were written with the help of Codex.

## 1. Folder Layout & Roles

```
cvintro/
├── POC_Dataset/          # Dataset root (see below for class layout)
├── train.py              # Main training/eval script with full arg parsing
├── Dataset.py            # Deterministic splits + transforms + dataloaders
├── Model.py              # ResNet18/34/50 builder, classifier head swap
├── run_train.sh          # Queue-based multi-GPU experiment launcher
├── result/               # Default checkpoints/metrics output (per run)
├── wandb/                # wandb run artifacts (optional)
└── README.md
```

Dataset structure expected by default:

```
cvintro/POC_Dataset/
├── Training/
│   ├── Chorionic_villi/
│   ├── Decidual_tissue/
│   ├── Hemorrhage/
│   └── Trophoblastic_tissue/
└── Testing/
    └── ...same class layout...
```

## 2. How to Run

Prerequisites: Python 3.8+, PyTorch/torchvision, tqdm, scikit-learn, Pillow, (optional) wandb. Install example:

```bash
pip install torch torchvision tqdm scikit-learn pillow wandb
```

Single-run example:

```bash
python /home/jaewonlee/cvintro/train.py \
  --data_root /home/jaewonlee/cvintro/POC_Dataset \
  --arch resnet34 \
  --batch_size 32 \
  --epochs 50 \
  --optimizer sgd \
  --learning_rate 0.01 \
  --weight_decay 1e-4 \
  --augmentations random_resized_crop horizontal_flip color_jitter rotation \
  --mixup_alpha 0.8 --mixup_prob 0.5 \
  --use_wandb
```

## 3. Key Arguments (similar to the original set)

- Data  
  - `--data_root`: Dataset root (default `POC_Dataset`).  
  - `--val_ratio`: Train/val split ratio (default 0.1, seed-fixed).  
  - `--image_size`: Input resolution (default 224).  
  - `--augmentations`: Train-time ops; choose from `random_resized_crop`, `horizontal_flip`, `vertical_flip`, `color_jitter`, `rotation`, `gaussian_blur`, `random_grayscale`, `center_crop`, `none`, ...  
  - `--eval_transforms`: Val/test transforms (default `center_crop`; use `resize`-only by omitting it).  
  - `--num_workers`, `--pin_memory`, `--drop_last` for DataLoader tuning.

- Model & Optimization  
  - `--arch`: `resnet18|resnet34|resnet50`.  
  - `--optimizer`: `sgd|adam|adamw`; momentum/betas/weight decay exposed.  
  - `--scheduler`: `none|multistep|cosine` with `--warmup_epochs` and `--milestones`.  
  - `--learning_rate`, `--epochs`, `--weight_decay`, `--batch_size`, `--dropout`.  

- Regularization & Precision  
  - `--label_smoothing`, `--max_grad_norm`, `--grad_accumulation_steps`.  
  - Mixup/CutMix: `--mixup_alpha`, `--cutmix_alpha`, `--mixup_prob`.  
  - `--random_erasing_prob` to enable RandomErasing.  
  - `--use_amp` for CUDA AMP.

- Logging / Checkpoints  
  - `--output_dir`: Base save dir (default `result`). Saves `best.pth`, `last.pth`, optional `epoch_*.pth` if `--save_every` > 0.  
  - wandb: `--use_wandb`, `--wandb_project`, `--wandb_run_name`, `--wandb_entity`.  
  - `--run_name` affects subfolder naming inside `output_dir`.  
  - `--evaluate_only` skips training and reports val/test metrics from a checkpoint (`--resume_path`).

- Misc  
  - `--seed`, `--deterministic` for reproducibility.  
  - `--device` or `--gpu` to pin a specific CUDA device.

## 4. End-to-End Pipeline

1) Dataset prep  
   - Place images under `POC_Dataset/Training` and `POC_Dataset/Testing` with class-wise folders.  
   - The script filters broken images, then creates a deterministic train/val split using `--val_ratio` and `--seed`.

2) Dataloaders & transforms  
   - Train transforms are built from `--augmentations` plus normalization and optional RandomErasing.  
   - Val/Test transforms default to center-crop + normalization (or resize-only if specified).

3) Model build  
   - Constructs ResNet18/34/50, replaces the final FC with a head sized to `--num_classes`, optional dropout, optional ImageNet pretraining.

4) Optimization loop  
   - Supports SGD/Adam/AdamW, cosine or multistep schedulers with warmup, gradient clipping, mixed precision, and grad accumulation.  
   - Optional Mixup/CutMix applied probabilistically; label smoothing handled in the loss.

5) Logging & checkpointing  
   - Prints tqdm progress; optionally logs to wandb.  
   - Saves `last.pth` each epoch, `best.pth` on improved val accuracy, and periodic `epoch_*.pth` if `--save_every` > 0.

6) Evaluation  
   - Runs full metrics (loss/acc/precision/recall/F1) on val and test loaders.  
   - `--evaluate_only` loads `--resume_path` and skips training.

## 5. Results
Experimental Results Summary

---

### Experiment 1: Learning Rate Search

Compare ResNet50 performance across different learning rates. The baseline is `LR = 0.02`.

- Comparison: `LR = 0.01` vs `0.02` vs `0.05` vs `0.10`
- Observation: When LR is too large (`0.10`), validation performance degrades. The range `0.02–0.05` appears to be a reasonable choice.

| Run Name      | LR    | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|---------------|-------|-----------|---------|----------|----------|----------------|-------------|
| r50_lr001     | 0.01  | 0.86357   | 0.9225  | 0.82321  | 0.81843  | 0.82728        | 0.81657     |
| r50_baseline  | 0.02  | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |
| r50_lr005     | 0.05  | 0.85552   | 0.9200  | 0.83618  | 0.83063  | 0.84014        | 0.82976     |
| r50_lr01      | 0.10  | 0.86079   | 0.8925  | 0.82730  | 0.82420  | 0.83047        | 0.82200     |

---

### Experiment 2: Optimizer Comparison

Compare traditional SGD (with momentum) and AdamW for ResNet50.

- Comparison: SGD (`LR = 0.02`) vs AdamW (`LR = 0.001`)
- Observation: AdamW shows higher test performance (`Test Acc = 0.84710`) than SGD, despite lower training accuracy, indicating better generalization.

| Run Name   | Optimizer | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|------------|-----------|-----------|---------|----------|----------|----------------|-------------|
| r50_sgd    | SGD       | 0.93748   | 0.9225  | 0.83754  | 0.83293  | 0.84302        | 0.83060     |
| r50_adamw  | AdamW     | 0.87969   | 0.9325  | 0.84710  | 0.84067  | 0.85185        | 0.83880     |

---

### Experiment 3: Weight Decay Search

Investigate the effect of different weight decay strengths on ResNet50.

- Comparison: `1e-5` (Weak) vs `1e-4` (Baseline) vs `5e-4` (Strong)
- Observation: The difference between `1e-4` and `5e-4` is small. In some metrics, `1e-5` even slightly outperforms the baseline.

| Run Name      | Weight Decay | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|---------------|--------------|-----------|---------|----------|----------|----------------|-------------|
| r50_wd1e5     | 1e-5         | 0.85774   | 0.9300  | 0.83140  | 0.82647  | 0.83502        | 0.82489     |
| r50_baseline  | 1e-4         | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |
| r50_wd5e4     | 5e-4         | 0.86246   | 0.9225  | 0.82799  | 0.82184  | 0.83413        | 0.81956     |

---

### Experiment 4: Regularization Combinations

Evaluate modern regularization techniques such as Label Smoothing, CutMix, and Mixup+CutMix.

- Comparison: No Regularization vs Label Smoothing vs CutMix vs Mixup+CutMix
- Observation: The no-regularization model has very high train accuracy (`0.95888`) but lower test accuracy (`0.83345`), suggesting overfitting. The `Mixup+CutMix` setting achieves the best test accuracy (`0.85051`) and F1, indicating better generalization.

| Run Name     | Method               | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|--------------|----------------------|-----------|---------|----------|----------|----------------|-------------|
| r50_no_reg   | None                 | 0.95888   | 0.9300  | 0.83345  | 0.82890  | 0.83949        | 0.82578     |
| r50_ls02     | LabelSmooth(0.2)     | 0.85968   | 0.9175  | 0.84164  | 0.83578  | 0.84689        | 0.83482     |
| r50_cutmix   | CutMix               | 0.85774   | 0.9175  | 0.84505  | 0.83871  | 0.85102        | 0.83756     |
| r50_mix_cut  | Mixup+CutMix         | 0.78494   | 0.9125  | 0.85051  | 0.84624  | 0.85987        | 0.84310     |

---

### Experiment 5: Augmentation Strength

Study the effect of data augmentation strength on model performance.

- Comparison: Light (Flip+Crop) vs Medium (Baseline, includes ColorJitter, etc.) vs Heavy (Blur, Erase, etc.)
- Observation: Light augmentation leads to significantly lower test performance (`Test Acc = 0.72765`). Medium or Heavy augmentation appears necessary for good generalization.

| Run Name        | Aug Level | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|-----------------|-----------|-----------|---------|----------|----------|----------------|-------------|
| r50_aug_light   | Light     | 0.87580   | 0.9275  | 0.72765  | 0.71124  | 0.71414        | 0.71538     |
| r50_baseline    | Medium    | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |
| r50_aug_heavy   | Heavy     | 0.88275   | 0.9375  | 0.84232  | 0.83597  | 0.84826        | 0.83368     |

---

### Experiment 6: Dropout Effect

Analyze the effect of inserting a Dropout layer before the final FC layer in ResNet50.

- Comparison: Dropout `0.0` (Baseline) vs `0.2` vs `0.5`
- Observation: Dropout `0.2` improves test accuracy (`0.84369`) over the baseline. Dropout `0.5` slightly improves over baseline but is worse than `0.2`.

| Run Name      | Dropout | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|---------------|---------|-----------|---------|----------|----------|----------------|-------------|
| r50_baseline  | 0.0     | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |
| r50_drop02    | 0.2     | 0.84885   | 0.9275  | 0.84369  | 0.83969  | 0.84807        | 0.82200     |
| r50_drop05    | 0.5     | 0.81578   | 0.9200  | 0.83413  | 0.82721  | 0.83971        | 0.82611     |

---

### Experiment 7: Model Architecture Comparison

Compare different ResNet architectures under the same training setup.

- Comparison: ResNet18 vs ResNet34 vs ResNet50
- Observation: ResNet18 outperforms ResNet50 on this dataset (`Test Acc = 0.84505` vs `0.82389`). The dataset size may be relatively small for ResNet50, leading to overfitting or suboptimal optimization.

| Run Name       | Arch     | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|----------------|----------|-----------|---------|----------|----------|----------------|-------------|
| r18_baseline   | ResNet18 | 0.87580   | 0.9275  | 0.84505  | 0.83894  | 0.85220        | 0.83632     |
| r34_baseline   | ResNet34 | 0.85496   | 0.9325  | 0.82321  | 0.81843  | 0.82728        | 0.81657     |
| r50_baseline   | ResNet50 | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |

---

### Experiment 8: Batch Size + LR Scaling

Apply the Linear Scaling Rule: scale learning rate linearly with batch size.

- Comparison: Batch 32 (`LR = 0.01`) vs Batch 64 (`LR = 0.02`, Baseline) vs Batch 128 (`LR = 0.04`)
- Observation: The `Batch 128 / LR 0.04` configuration achieves the best test accuracy (`0.85324`), the highest among all experiments.

| Run Name        | Batch / LR | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|-----------------|------------|-----------|---------|----------|----------|----------------|-------------|
| r50_b32_lr01    | B32 / 0.01 | 0.81134   | 0.9350  | 0.83003  | 0.82430  | 0.83846        | 0.82116     |
| r50_baseline    | B64 / 0.02 | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |
| r50_b128_lr04   | B128 / 0.04| 0.86885   | 0.9250  | 0.85324  | 0.84695  | 0.86186        | 0.84506     |

---

