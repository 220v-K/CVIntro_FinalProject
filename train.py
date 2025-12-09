import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.cuda import amp
from tqdm import tqdm

from Dataset import create_dataloaders
from Model import available_models, build_model

try:
    import wandb
except ImportError:
    wandb = None


# 재현성을 위해 모든 난수 시드를 고정한다.
def setup_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()

    ########################################
    # Data 설정
    ########################################
    parser.add_argument("--data_root", type=str, default="POC_Dataset")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=["random_resized_crop", "horizontal_flip", "color_jitter"],
        help="학습 시 사용할 augmentation 목록",
    )
    parser.add_argument(
        "--eval_transforms",
        nargs="+",
        default=["center_crop"],
        help="검증 및 테스트 시 사용할 변환 옵션",
    )
    parser.add_argument("--random_resized_crop_scale", type=float, default=0.8)
    parser.add_argument("--color_jitter_strength", type=float, default=0.2)
    parser.add_argument("--rotation_degree", type=float, default=15.0)
    parser.add_argument("--gaussian_kernel", type=int, default=3)
    parser.add_argument("--gaussian_sigma", type=float, default=1.0)
    parser.add_argument("--random_erasing_prob", type=float, default=0.0)
    parser.add_argument(
        "--normalize_mean",
        type=float,
        nargs=3,
        default=(0.485, 0.456, 0.406),
    )
    parser.add_argument(
        "--normalize_std",
        type=float,
        nargs=3,
        default=(0.229, 0.224, 0.225),
    )

    ########################################
    # Model 설정
    ########################################
    parser.add_argument("--arch", type=str, default="resnet18", choices=available_models())
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    ########################################
    # Optimization 설정
    ########################################
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "adam", "adamw"],
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "multistep", "cosine"])
    parser.add_argument("--milestones", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--cutmix_alpha", type=float, default=0.0)
    parser.add_argument("--mixup_prob", type=float, default=0.0)
    parser.add_argument("--use_amp", action="store_true")

    ########################################
    # Logging 및 기타 옵션
    ########################################
    parser.add_argument("--output_dir", type=str, default="result")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", help="예: cuda, cpu, cuda:0 등")
    parser.add_argument("--gpu", type=int, default=None, help="지정 시 device를 cuda:<gpu>로 설정")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="POC_Dataset")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_name", dest="wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="poc_resnet")

    return parser.parse_args()


def build_optimizer(model: nn.Module, args):
    """
    다양한 optimizer 선택지를 제공한다.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    if args.optimizer == "adam":
        return torch.optim.Adam(
            params,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )


def compute_scheduled_lr(epoch: int, args) -> float:
    """
    Warmup과 Cosine/MultiStep 스케줄을 단순 수식으로 계산한다.
    """
    base_lr = args.learning_rate
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        warmup_factor = (epoch + 1) / max(1, args.warmup_epochs)
        return base_lr * warmup_factor

    effective_epoch = epoch - args.warmup_epochs
    total_main_epochs = max(1, args.epochs - args.warmup_epochs)

    if args.scheduler == "cosine":
        progress = min(1.0, effective_epoch / total_main_epochs)
        return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
    if args.scheduler == "multistep":
        lr = base_lr
        for milestone in sorted(args.milestones):
            if epoch >= milestone:
                lr *= args.scheduler_gamma
        return lr
    return base_lr


def adjust_learning_rate(optimizer, epoch, args):
    lr = compute_scheduled_lr(epoch, args)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)


def rand_bbox(size, lam):
    _, _, h, w = size
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = random.randint(0, w - 1)
    cy = random.randint(0, h - 1)

    bbx1 = int(np.clip(cx - cut_w // 2, 0, w))
    bby1 = int(np.clip(cy - cut_h // 2, 0, h))
    bbx2 = int(np.clip(cx + cut_w // 2, 0, w))
    bby2 = int(np.clip(cy + cut_h // 2, 0, h))
    return bbx1, bby1, bbx2, bby2


def apply_mixup_cutmix(images, labels, args):
    """
    mixup/cutmix 중 하나를 확률적으로 적용한다.
    """
    if args.mixup_alpha <= 0 and args.cutmix_alpha <= 0:
        return images, None

    mix_selector = random.random()
    use_cutmix = False
    if args.cutmix_alpha > 0:
        if args.mixup_alpha > 0:
            use_cutmix = mix_selector < 0.5
        else:
            use_cutmix = True

    if use_cutmix:
        lam = np.random.beta(args.cutmix_alpha, args.cutmix_alpha)
        rand_index = torch.randperm(images.size(0), device=images.device)
        shuffled = images[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bby1:bby2, bbx1:bbx2] = shuffled[:, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bby2 - bby1) * (bbx2 - bbx1) / (images.size(-1) * images.size(-2)))
        labels_a = labels
        labels_b = labels[rand_index]
        return images, (labels_a, labels_b, lam)

    if args.mixup_alpha > 0:
        lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
        rand_index = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1 - lam) * images[rand_index]
        labels_a = labels
        labels_b = labels[rand_index]
        return mixed, (labels_a, labels_b, lam)

    return images, None


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    """
    한 epoch 동안 학습을 진행하며 손실과 정확도를 반환한다.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    sample_count = 0

    progress = tqdm(loader, desc=f"Train [{epoch + 1}/{args.epochs}]", leave=False)
    optimizer.zero_grad()
    for step, (images, labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_for_acc = labels.detach()

        use_mix = args.mixup_prob > 0 and random.random() < args.mixup_prob
        mix_targets = None
        if use_mix:
            images, mix_targets = apply_mixup_cutmix(images, labels, args)
            if mix_targets is None:
                use_mix = False

        with amp.autocast(enabled=args.use_amp and device.type == "cuda"):
            outputs = model(images)
            if use_mix and mix_targets is not None:
                targets_a, targets_b, lam = mix_targets
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)

        loss_value = loss.item()
        loss = loss / args.grad_accumulation_steps
        if args.use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.grad_accumulation_steps == 0 or step == len(loader):
            if args.max_grad_norm > 0:
                if args.use_amp and device.type == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.use_amp and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels_for_acc).sum().item()
        sample_count += labels.size(0)
        running_loss += loss_value * labels.size(0)

        avg_loss = running_loss / max(1, sample_count)
        avg_acc = running_correct / max(1, sample_count)
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    epoch_loss = running_loss / max(1, sample_count)
    epoch_acc = running_correct / max(1, sample_count)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Eval"):
    """
    검증/테스트 데이터셋에서 모델 성능을 측정
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    sample_count = 0
    all_preds = []
    all_targets = []

    progress = tqdm(loader, desc=desc, leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * labels.size(0)
        running_correct += (preds == labels).sum().item()
        sample_count += labels.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())

        avg_loss = running_loss / max(1, sample_count)
        avg_acc = running_correct / max(1, sample_count)
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    epoch_loss = running_loss / max(1, sample_count)
    epoch_acc = running_correct / max(1, sample_count)

    if sample_count > 0:
        preds_concat = torch.cat(all_preds)
        targets_concat = torch.cat(all_targets)
        # 다중 클래스 분류의 전반적인 균형을 보기 위해 macro average 사용
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_concat.numpy(),
            preds_concat.numpy(),
            average="macro",
            zero_division=0,
        )
    else:
        precision = recall = f1 = 0.0

    return epoch_loss, epoch_acc, precision, recall, f1


def save_checkpoint(state, path: Path):
    """
    학습 상태를 지정한 경로에 저장한다.
    """
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, device):
    """
    저장된 체크포인트를 불러와 모델과 옵티마이저를 복원한다.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_val_acc", 0.0)


def main():
    args = parse_args()
    args.normalize_mean = tuple(args.normalize_mean)
    args.normalize_std = tuple(args.normalize_std)
    args.milestones = sorted(args.milestones)
    if args.grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps 값은 1 이상의 정수여야 합니다.")

    if args.gpu is not None:
        device_str = f"cuda:{args.gpu}"
    else:
        device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    setup_seed(args.seed, args.deterministic)

    run_dir_name = args.wandb_run_name or args.run_name
    output_dir = Path(args.output_dir) / run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save_dir] {output_dir.resolve()}")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(args)

    model = build_model(
        arch=args.arch,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = build_optimizer(model, args)
    scaler = amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    start_epoch = 0
    best_val_acc = 0.0
    resume_path = args.resume_path
    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            print(f"Resume training from {resume_path}")
            start_epoch, best_val_acc = load_checkpoint(model, optimizer, resume_path, device)
        else:
            print(f"Checkpoint not found: {resume_path}")

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb 라이브러리가 설치되어 있지 않습니다.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )

    if args.evaluate_only:
        print("평가 모드로 전환합니다.")
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, criterion, device, desc="Val"
        )
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
            model, test_loader, criterion, device, desc="Test"
        )
        print(
            f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
            f"P: {val_prec:.4f}, R: {val_rec:.4f}, F1: {val_f1:.4f}"
        )
        print(
            f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
            f"P: {test_prec:.4f}, R: {test_rec:.4f}, F1: {test_f1:.4f}"
        )
        if wandb_run:
            wandb.log(
                {
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/precision": val_prec,
                    "val/recall": val_rec,
                    "val/f1": val_f1,
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/precision": test_prec,
                    "test/recall": test_rec,
                    "test/f1": test_f1,
                }
            )
            wandb_run.finish()
        return

    for epoch in range(start_epoch, args.epochs):
        current_lr = adjust_learning_rate(optimizer, epoch, args)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args
        )
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, criterion, device, desc="Val"
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        save_dict = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": class_names,
            "args": vars(args),
        }
        save_checkpoint(save_dict, output_dir / "last.pth")
        if is_best:
            save_checkpoint(save_dict, output_dir / "best.pth")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(save_dict, output_dir / f"epoch_{epoch + 1}.pth")

        if wandb_run:
            wandb.log(
                {
                    "lr": current_lr,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/precision": val_prec,
                    "val/recall": val_rec,
                    "val/f1": val_f1,
                    "epoch": epoch + 1,
                }
            )

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Val P/R/F1: {val_prec:.4f}/{val_rec:.4f}/{val_f1:.4f}"
        )

    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, criterion, device, desc="Test"
    )
    print(f"Best Validation Acc: {best_val_acc:.4f}")
    print(
        f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
        f"P: {test_prec:.4f}, R: {test_rec:.4f}, F1: {test_f1:.4f}"
    )
    if wandb_run:
        wandb.log(
            {
                "test/loss": test_loss,
                "test/acc": test_acc,
                "test/precision": test_prec,
                "test/recall": test_rec,
                "test/f1": test_f1,
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
