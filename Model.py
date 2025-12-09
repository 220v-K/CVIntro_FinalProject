import torch.nn as nn
import torchvision.models as tv_models


# 사용할 수 있는 ResNet 계열 모델들을 미리 정의해둔다.
MODEL_FACTORY = {
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
}

WEIGHT_ENUMS = {
    "resnet18": getattr(tv_models, "ResNet18_Weights", None),
    "resnet34": getattr(tv_models, "ResNet34_Weights", None),
    "resnet50": getattr(tv_models, "ResNet50_Weights", None),
}


def build_model(arch: str, num_classes: int, pretrained: bool = False, dropout: float = 0.0) -> nn.Module:
    """
    지정한 ResNet 아키텍처를 생성하고, 분류기 부분을 현재 과제에 맞게 교체한다.
    """
    arch = arch.lower()
    if arch not in MODEL_FACTORY:
        raise ValueError(f"지원하지 않는 모델입니다: {arch}")

    try:
        weight_enum = WEIGHT_ENUMS.get(arch)
        weights = None
        if pretrained and weight_enum is not None:
            weights = weight_enum.IMAGENET1K_V1
        model = MODEL_FACTORY[arch](weights=weights)
    except (TypeError, AttributeError):
        # 구버전 호환을 위해 pretrained 플래그를 그대로 전달한다.
        model = MODEL_FACTORY[arch](pretrained=pretrained)

    in_features = model.fc.in_features
    classifier: nn.Module
    if dropout > 0:
        classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        classifier = nn.Linear(in_features, num_classes)

    model.fc = classifier
    return model


def available_models():
    """
    사용 가능한 모델 이름 목록을 반환하여 argparse 등의 도움말에 활용한다.
    """
    return sorted(MODEL_FACTORY.keys())


__all__ = ["build_model", "available_models"]
