from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# Validation split에도 독자적인 변환을 적용할 수 있도록 감싸는 Dataset 구현.
class TransformSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: Sequence[int], transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _split_indices(num_samples: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """
    항상 동일한 데이터 분할을 위해 seed 기반으로 index를 나눈다.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio는 0과 1 사이의 값이어야 합니다.")
    if num_samples < 2:
        raise ValueError("Validation 분할을 수행하기에 데이터 수가 부족합니다.")

    val_size = max(1, int(round(num_samples * val_ratio)))
    if val_size >= num_samples:
        val_size = num_samples - 1

    generator = torch.Generator().manual_seed(seed)
    permuted = torch.randperm(num_samples, generator=generator).tolist()
    val_indices = permuted[:val_size]
    train_indices = permuted[val_size:]
    return train_indices, val_indices


def _build_augmentation(name: str, args):
    """
    argparse 에서 전달받은 문자열 이름을 torchvision 변환으로 바꾼다.
    """
    name = name.lower()
    if name == "random_resized_crop":
        return transforms.RandomResizedCrop(
            args.image_size, scale=(args.random_resized_crop_scale, 1.0)
        )
    if name == "horizontal_flip":
        return transforms.RandomHorizontalFlip()
    if name == "vertical_flip":
        return transforms.RandomVerticalFlip()
    if name == "color_jitter":
        return transforms.ColorJitter(
            brightness=args.color_jitter_strength,
            contrast=args.color_jitter_strength,
            saturation=args.color_jitter_strength,
            hue=min(0.5, args.color_jitter_strength / 2),
        )
    if name == "rotation":
        return transforms.RandomRotation(args.rotation_degree)
    if name == "gaussian_blur":
        kernel = args.gaussian_kernel if args.gaussian_kernel % 2 == 1 else args.gaussian_kernel + 1
        return transforms.GaussianBlur(kernel_size=kernel, sigma=args.gaussian_sigma)
    if name == "random_grayscale":
        return transforms.RandomGrayscale(p=0.2)
    if name == "center_crop":
        return transforms.CenterCrop(args.image_size)
    if name == "none":
        return None
    raise ValueError(f"알 수 없는 augmentation 옵션: {name}")


def _build_transforms(train: bool, args):
    """
    학습/검증 단계별로 필요한 전체 변환 파이프라인을 구성한다.
    """
    normalize = transforms.Normalize(
        mean=getattr(args, "normalize_mean", (0.485, 0.456, 0.406)),
        std=getattr(args, "normalize_std", (0.229, 0.224, 0.225)),
    )
    ops: List = []

    if train:
        # RandomResizedCrop가 명시되면 Resize보다 먼저 처리하고, 중복 적용을 막는다.
        if "random_resized_crop" in args.augmentations:
            ops.append(_build_augmentation("random_resized_crop", args))
        else:
            ops.extend(
                [
                    transforms.Resize((args.image_size, args.image_size)),
                ]
            )
        for aug_name in args.augmentations:
            if aug_name == "random_resized_crop":
                continue
            aug = _build_augmentation(aug_name, args)
            if aug is None:
                continue
            ops.append(aug)
    else:
        eval_ops = getattr(args, "eval_transforms", ["center_crop"])
        if "center_crop" in eval_ops:
            ops.append(transforms.Resize(int(args.image_size * 1.1)))
            ops.append(transforms.CenterCrop(args.image_size))
        else:
            ops.append(transforms.Resize((args.image_size, args.image_size)))

    ops.append(transforms.ToTensor())
    ops.append(normalize)

    if train and getattr(args, "random_erasing_prob", 0.0) > 0:
        ops.append(
            transforms.RandomErasing(
                p=getattr(args, "random_erasing_prob", 0.0),
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
            )
        )
    return transforms.Compose(ops)


def _filter_broken_samples(dataset: datasets.ImageFolder) -> int:
    """
    ImageFolder 샘플 중 깨진 이미지를 제외하고 in-place로 수정한다.
    반환값은 제거된 샘플 개수.
    """
    filtered_samples = []
    filtered_targets = []
    removed = 0
    for path, target in dataset.samples:
        try:
            with Image.open(path) as img:
                img.verify()  # 빠르게 헤더 검증
        except Exception:
            removed += 1
            continue
        filtered_samples.append((path, target))
        filtered_targets.append(target)

    dataset.samples = filtered_samples
    dataset.imgs = filtered_samples
    dataset.targets = filtered_targets
    return removed


def create_dataloaders(args):
    """
    Training/Validation/Test DataLoader를 한 번에 구성하는 헬퍼 함수.
    """
    root = Path(args.data_root)
    train_dir = root / "Training"
    test_dir = root / "Testing"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training 폴더가 존재하지 않습니다: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Testing 폴더가 존재하지 않습니다: {test_dir}")

    base_train_dataset = datasets.ImageFolder(train_dir)
    removed_train = _filter_broken_samples(base_train_dataset)
    train_indices, val_indices = _split_indices(
        len(base_train_dataset), args.val_ratio, args.seed
    )

    train_transform = _build_transforms(train=True, args=args)
    eval_transform = _build_transforms(train=False, args=args)

    train_dataset = TransformSubset(
        base_train_dataset, train_indices, transform=train_transform
    )
    val_dataset = TransformSubset(
        base_train_dataset, val_indices, transform=eval_transform
    )
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    removed_test = _filter_broken_samples(test_dataset)
    if removed_train or removed_test:
        print(
            f"[dataset] removed broken images -> train: {removed_train}, test: {removed_test}"
        )

    common_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=args.drop_last,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, val_loader, test_loader, base_train_dataset.classes


__all__ = ["create_dataloaders"]
