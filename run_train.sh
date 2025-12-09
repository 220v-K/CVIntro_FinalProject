#!/usr/bin/env bash
# gpu_wait_run_queue.sh : 지정한 GPU 집합에서 "순차적으로" 커맨드들을 실행
# - 1번 커맨드가 끝나면 빈 GPU를 찾아 2번 커맨드 실행 … (순차 처리)
# - 각 커맨드는 공백 안전(배열)으로 보관

set -euo pipefail
# 디버그가 필요하면 주석 해제
# set -x

# 확인할 GPU 목록과 루프 간격(초)
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
SLEEP=${SLEEP:-120}

# 선택 GPU만 보이게 할지 여부(1이면 CUDA_VISIBLE_DEVICES=g, --gpu 0으로 실행)
ONLY_VISIBLE=${ONLY_VISIBLE:-0}

# 공통 기본 옵션
BASE_OPTS=(
  python train.py
  --data_root /home/jaewonlee/cvintro/POC_Dataset
  --epochs 60
  --scheduler cosine
  --use_wandb
  --use_amp
  --seed 42
)

###############################################################################
# 실험 1: Learning Rate 탐색 (ResNet50 기준)
###############################################################################
declare -a CMD_R50_LR001=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.01 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_lr001" --wandb_run_name "r50_lr001"
)

declare -a CMD_R50_LR005=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.05 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_lr005" --wandb_run_name "r50_lr005"
)

declare -a CMD_R50_LR01=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.1 --weight_decay 1e-4 --warmup_epochs 5
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_lr01" --wandb_run_name "r50_lr01"
)

###############################################################################
# 실험 2: Optimizer 비교 (SGD vs AdamW)
###############################################################################
declare -a CMD_R50_SGD=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3 --momentum 0.9
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_sgd" --wandb_run_name "r50_sgd"
)

declare -a CMD_R50_ADAMW=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer adamw
  --learning_rate 1e-3 --weight_decay 0.05 --warmup_epochs 5
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_adamw" --wandb_run_name "r50_adamw"
)

###############################################################################
# 실험 3: Weight Decay 탐색
###############################################################################
declare -a CMD_R50_WD1E5=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-5 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_wd1e5" --wandb_run_name "r50_wd1e5"
)

declare -a CMD_R50_WD5E4=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 5e-4 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_wd5e4" --wandb_run_name "r50_wd5e4"
)

###############################################################################
# 실험 4: Regularization 조합 (Label Smoothing + Mixup/CutMix)
###############################################################################
declare -a CMD_R50_NO_REG=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.0 --mixup_alpha 0.0 --mixup_prob 0.0
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_no_reg" --wandb_run_name "r50_no_reg"
)

declare -a CMD_R50_LS02=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.2 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_ls02" --wandb_run_name "r50_ls02"
)

declare -a CMD_R50_CUTMIX=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.1 --cutmix_alpha 1.0 --mixup_prob 0.5
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_cutmix" --wandb_run_name "r50_cutmix"
)

declare -a CMD_R50_MIX_CUT=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.4 --cutmix_alpha 1.0 --mixup_prob 0.5
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_mix_cut" --wandb_run_name "r50_mix_cut"
)

###############################################################################
# 실험 5: Augmentation 강도 비교
###############################################################################
declare -a CMD_R50_AUG_LIGHT=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.0 --mixup_prob 0.0
  --augmentations random_resized_crop horizontal_flip
  --run_name "r50_aug_light" --wandb_run_name "r50_aug_light"
)

declare -a CMD_R50_AUG_HEAVY=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation gaussian_blur
  --random_erasing_prob 0.25
  --run_name "r50_aug_heavy" --wandb_run_name "r50_aug_heavy"
)

###############################################################################
# 실험 6: Dropout 효과
###############################################################################
declare -a CMD_R50_DROP02=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3 --dropout 0.2
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_drop02" --wandb_run_name "r50_drop02"
)

declare -a CMD_R50_DROP05=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3 --dropout 0.5
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_drop05" --wandb_run_name "r50_drop05"
)

###############################################################################
# 실험 7: 모델 아키텍처 비교 (동일 설정)
###############################################################################
declare -a CMD_R18_BASELINE=(
  "${BASE_OPTS[@]}"
  --arch resnet18 --batch_size 96 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r18_baseline" --wandb_run_name "r18_baseline"
)

declare -a CMD_R34_BASELINE=(
  "${BASE_OPTS[@]}"
  --arch resnet34 --batch_size 96 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r34_baseline" --wandb_run_name "r34_baseline"
)

declare -a CMD_R50_BASELINE=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 64 --optimizer sgd
  --learning_rate 0.02 --weight_decay 1e-4 --warmup_epochs 3
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_baseline" --wandb_run_name "r50_baseline"
)

###############################################################################
# 실험 8: Batch Size + LR 스케일링 (Linear Scaling Rule)
###############################################################################
declare -a CMD_R50_B32_LR01=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 32 --optimizer sgd
  --learning_rate 0.01 --weight_decay 1e-4 --warmup_epochs 5
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_b32_lr01" --wandb_run_name "r50_b32_lr01"
)

declare -a CMD_R50_B128_LR04=(
  "${BASE_OPTS[@]}"
  --arch resnet50 --batch_size 128 --optimizer sgd
  --learning_rate 0.04 --weight_decay 1e-4 --warmup_epochs 5
  --label_smoothing 0.1 --mixup_alpha 0.4 --mixup_prob 0.3
  --augmentations random_resized_crop horizontal_flip color_jitter rotation
  --run_name "r50_b128_lr04" --wandb_run_name "r50_b128_lr04"
)

###############################################################################
# 순서대로 실행할 목록 (원하는 실험만 선택)
###############################################################################
CMDS=(
  # Learning Rate 탐색
  CMD_R50_LR001
  CMD_R50_LR005
  CMD_R50_LR01
  # Optimizer 비교
#   CMD_R50_SGD
#   CMD_R50_ADAMW
#   # Weight Decay 탐색
#   CMD_R50_WD1E5
#   CMD_R50_WD5E4
  # Regularization 조합
#   CMD_R50_NO_REG
#   CMD_R50_LS02
#   CMD_R50_CUTMIX
#   CMD_R50_MIX_CUT
  # Augmentation 강도
#   CMD_R50_AUG_LIGHT
#   CMD_R50_AUG_HEAVY
#   # Dropout 효과
#   CMD_R50_DROP02
#   CMD_R50_DROP05
  # 모델 아키텍처 비교
#   CMD_R18_BASELINE
#   CMD_R34_BASELINE
#   CMD_R50_BASELINE
#   # Batch Size + LR 스케일링
#   CMD_R50_B32_LR01
#   CMD_R50_B128_LR04
)

wait_for_idle_gpu() {
  while true; do
    local any_busy=false
    for g in "${GPUS[@]}"; do
      # nvidia-smi 실패해도 루프 계속
      local running
      running=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null || true)

      # 일부 드라이버는 "No running processes found"를 출력하므로 그 경우도 idle로 처리
      if [[ -z "${running//[[:space:]]/}" ]] || [[ "$running" =~ [Nn]o[[:space:]]+running ]]; then
        # stdout: GPU 번호만
        printf '%s' "$g"
        return 0
      else
        any_busy=true
        # stderr: 대기 로그
        >&2 echo "$(date '+%F %T') : GPU $g busy (PID=${running//$'\n'/,})"
      fi
    done
    if [ "$any_busy" = true ]; then
      >&2 echo "$(date '+%F %T') : all requested GPUs busy → retry in ${SLEEP}s"
      sleep "$SLEEP"
    fi
  done
}

# 순차 실행
for cname in "${CMDS[@]}"; do
  # 배열 이름(cname)을 실제 배열로 전개(공백 안전)
  eval 'CMD_TO_RUN=("${'"$cname"'[@]}")'

  g="$(wait_for_idle_gpu)"
  echo "$(date '+%F %T') : GPU $g idle → run $cname"

  if [ "${ONLY_VISIBLE:-0}" = 1 ]; then
    CUDA_VISIBLE_DEVICES="$g" "${CMD_TO_RUN[@]}" --gpu 0
  else
    "${CMD_TO_RUN[@]}" --gpu "$g"
  fi
done

# for cname in "${CMDS[@]}"; do
#   # 배열 이름(cname)을 실제 배열로 전개(공백 안전)
#   eval 'CMD_TO_RUN=("${'"$cname"'[@]}")'

#   "${CMD_TO_RUN[@]}" --gpu 7
# done

echo "$(date '+%F %T') : all commands completed."
