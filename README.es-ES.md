# Pipeline de Clasificación de POC_Dataset con ResNet

> Todo el código fue escrito por mí, y solo algunas frases del README fueron escritas con la ayuda de Codex.

## 1. Diseño de Carpetas y Roles

```
cvintro/
├── POC_Dataset/          # Raíz del dataset (ver abajo la disposición de clases)
├── train.py              # Script principal de entrenamiento/eval con parsing completo de argumentos
├── Dataset.py            # Divisiones deterministas + transformaciones + dataloaders
├── Model.py              # Constructor de ResNet18/34/50, intercambio de cabezal del clasificador
├── run_train.sh          # Lanzador de experimentos multi-GPU basado en colas
├── result/               # Salida predeterminada de checkpoints/métricas (por ejecución)
├── wandb/                # Artefactos de ejecución de wandb (opcional)
└── README.md
```

Estructura de dataset esperada por defecto:

```
cvintro/POC_Dataset/
├── Training/
│   ├── Chorionic_villi/
│   ├── Decidual_tissue/
│   ├── Hemorrhage/
│   └── Trophoblastic_tissue/
└── Testing/
    └── ...misma disposición de clases...
```

## 2. Cómo Ejecutar

Prerrequisitos: Python 3.8+, PyTorch/torchvision, tqdm, scikit-learn, Pillow, (opcional) wandb. Ejemplo de instalación:

```bash
pip install torch torchvision tqdm scikit-learn pillow wandb
```

Ejemplo de ejecución única:

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

## 3. Argumentos Clave (similares al conjunto original)

- Datos  
  - `--data_root`: Raíz del dataset (por defecto `POC_Dataset`).  
  - `--val_ratio`: Proporción de división train/val (por defecto 0.1, semilla fija).  
  - `--image_size`: Resolución de entrada (por defecto 224).  
  - `--augmentations`: Operaciones de tiempo de entrenamiento; elegir entre `random_resized_crop`, `horizontal_flip`, `vertical_flip`, `color_jitter`, `rotation`, `gaussian_blur`, `random_grayscale`, `center_crop`, `none`, ...  
  - `--eval_transforms`: Transformaciones de val/test (por defecto `center_crop`; use solo `resize` omitiéndolo).  
  - `--num_workers`, `--pin_memory`, `--drop_last` para ajuste del DataLoader.

- Modelo y Optimización  
  - `--arch`: `resnet18|resnet34|resnet50`.  
  - `--optimizer`: `sgd|adam|adamw`; momentum/betas/weight decay expuestos.  
  - `--scheduler`: `none|multistep|cosine` con `--warmup_epochs` y `--milestones`.  
  - `--learning_rate`, `--epochs`, `--weight_decay`, `--batch_size`, `--dropout`.  

- Regularización y Precisión  
  - `--label_smoothing`, `--max_grad_norm`, `--grad_accumulation_steps`.  
  - Mixup/CutMix: `--mixup_alpha`, `--cutmix_alpha`, `--mixup_prob`.  
  - `--random_erasing_prob` para habilitar RandomErasing.  
  - `--use_amp` para CUDA AMP.

- Registro / Checkpoints  
  - `--output_dir`: Directorio base de guardado (por defecto `result`). Guarda `best.pth`, `last.pth`, y opcionalmente `epoch_*.pth` si `--save_every` > 0.  
  - wandb: `--use_wandb`, `--wandb_project`, `--wandb_run_name`, `--wandb_entity`.  
  - `--run_name` afecta el nombre de la subcarpeta dentro de `output_dir`.  
  - `--evaluate_only` omite el entrenamiento y reporta métricas de val/test desde un checkpoint (`--resume_path`).

- Misceláneos  
  - `--seed`, `--deterministic` para reproducibilidad.  
  - `--device` o `--gpu` para fijar un dispositivo CUDA específico.

## 4. Pipeline de Extremo a Extremo

1) Preparación del Dataset  
   - Colocar las imágenes bajo `POC_Dataset/Training` y `POC_Dataset/Testing` con carpetas por clase.  
   - El script filtra imágenes corruptas, luego crea una división train/val determinista usando `--val_ratio` y `--seed`.

2) Dataloaders y Transformaciones  
   - Las transformaciones de entrenamiento se construyen a partir de `--augmentations` más normalización y RandomErasing opcional.  
   - Las transformaciones de Val/Test son por defecto center-crop + normalización (o solo resize si se especifica).

3) Construcción del Modelo  
   - Construye ResNet18/34/50, reemplaza la FC final con un cabezal ajustado a `--num_classes`, dropout opcional.

4) Bucle de Optimización  
   - Soporta SGD/Adam/AdamW, schedulers cosine o multistep con warmup, recorte de gradientes, precisión mixta y acumulación de gradientes.  
   - Mixup/CutMix opcional aplicados probabilísticamente; el label smoothing se maneja en la pérdida.

5) Registro y Checkpointing  
   - Imprime el progreso con tqdm; opcionalmente registra en wandb.  
   - Guarda `last.pth` cada época, `best.pth` al mejorar la precisión de validación, y `epoch_*.pth` periódicamente si `--save_every` > 0.

6) Evaluación  
   - Ejecuta métricas completas (pérdida/precisión/precision/recall/F1) en los cargadores de val y test.  
   - `--evaluate_only` carga `--resume_path` y omite el entrenamiento.

## 5. Resultados
Resumen de Resultados Experimentales

---

### Experimento 1: Búsqueda de Tasa de Aprendizaje (LR)

Comparación del rendimiento de ResNet50 a través de diferentes tasas de aprendizaje. La línea base es `LR = 0.02`.

- Comparación: `LR = 0.01` vs `0.02` vs `0.05` vs `0.10`
- Observación: Cuando el LR es demasiado grande (`0.10`), el rendimiento de validación degrada. El rango `0.02–0.05` parece ser una elección razonable.

| Nombre de Ejecución | LR    | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|--------------------|-------|-----------|---------|----------|----------|----------------|-------------|
| r50_lr001          | 0.01  | 0.86357   | 0.9225  | 0.82321  | 0.81843  | 0.82728        | 0.81657     |
| r50_baseline      | 0.02  | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |
| r50_lr005          | 0.05  | 0.85552   | 0.9200  | 0.83618  | 0.83063  | 0.84014        | 0.82976     |
| r50_lr01           | 0.10  | 0.86079   | 0.8925  | 0.82730  | 0.82420  | 0.83047        | 0.82200     |

---

### Experimento 2: Comparación de Optimizadores

Comparación entre SGD tradicional (con momentum) y AdamW para ResNet50.

- Comparación: SGD (`LR = 0.02`) vs AdamW (`LR = 0.001`)
- Observación: AdamW muestra un mayor rendimiento en test (`Test Acc = 0.84710`) que SGD, a pesar de una menor precisión de entrenamiento, lo que indica una mejor generalización.

| Nombre de Ejecución | Optimizador | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|--------------------|-------------|-----------|---------|----------|----------|----------------|-------------|
| r50_sgd            | SGD         | 0.93748   | 0.9225  | 0.83754  | 0.83293  | 0.84302        | 0.83060     |
| r50_adamw          | AdamW       | 0.87969   | 0.9325  | 0.84710  | 0.84067  | 0.85185        | 0.83880     |

---

### Experimento 3: Búsqueda de Weight Decay

Investigación del efecto de diferentes intensidades de weight decay en ResNet50.

- Comparación: `1e-5` (Débil) vs `1e-4` (Línea base) vs `5e-4` (Fuerte)
- Observación: La diferencia entre `1e-4` y `5e-4` es pequeña. En algunas métricas, `1e-5` incluso supera ligeramente a la línea base.

| Nombre de Ejecución | Weight Decay | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|--------------------|--------------|-----------|---------|----------|----------|----------------|-------------|
| r50_wd1e5          | 1e-5         | 0.85774   | 0.9300  | 0.83140  | 0.82647  | 0.83502        | 0.82489     |
| r50_baseline       | 1e-4         | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |
| r50_wd5e4          | 5e-4         | 0.86246   | 0.9225  | 0.82799  | 0.82184  | 0.83413        | 0.81956     |

---

### Experimento 4: Combinaciones de Regularización

Evaluación de técnicas de regularización modernas como Label Smoothing, CutMix y Mixup+CutMix.

- Comparación: Sin Regularización vs Label Smoothing vs CutMix vs Mixup+CutMix
- Observación: El modelo sin regularización tiene una precisión de entrenamiento muy alta (`0.95888`) pero una precisión de test menor (`0.83345`), lo que sugiere sobreajuste (overfitting). La configuración `Mixup+CutMix` logra la mejor precisión de test (`0.85051`) y F1, indicando una mejor generalización.

| Nombre de Ejecución | Método               | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|--------------------|----------------------|-----------|---------|----------|----------|----------------|-------------|
| r50_no_reg         | Ninguno              | 0.95888   | 0.9300  | 0.83345  | 0.82890  | 0.83949        | 0.82578     |
| r50_ls02           | LabelSmooth(0.2)     | 0.85968   | 0.9175  | 0.84164  | 0.83578  | 0.84689        | 0.83482     |
| r50_cutmix         | CutMix               | 0.85774   | 0.9175  | 0.84505  | 0.83871  | 0.85102        | 0.83756     |
| r50_mix_cut        | Mixup+CutMix         | 0.78494   | 0.9125  | 0.85051  | 0.84624  | 0.85987        | 0.84310     |

---

### Experimento 5: Intensidad de la Aumentación

Estudio del efecto de la intensidad de la aumentación de datos en el rendimiento del modelo.

- Comparación: Ligera (Flip+Crop) vs Media (Línea base, incluye ColorJitter, etc.) vs Pesada (Blur, Erase, etc.)
- Observación: La aumentación ligera conduce a un rendimiento de test significativamente menor (`Test Acc = 0.72765`). La aumentación Media o Pesada parece necesaria para una buena generalización.

| Nombre de Ejecución | Nivel Aug | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|--------------------|-----------|-----------|---------|----------|----------|----------------|-------------|
| r50_aug_light      | Ligera    | 0.87580   | 0.9275  | 0.72765  | 0.71124  | 0.71414        | 0.71538     |
| r50_baseline       | Media     | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |
| r50_aug_heavy      | Pesada    | 0.88275   | 0.9375  | 0.84232  | 0.83597  | 0.84826        | 0.83368     |

---

### Experimento 6: Efecto del Dropout

Análisis del efecto de insertar una capa de Dropout antes de la capa FC final en ResNet50.

- Comparación: Dropout `0.0` (Línea base) vs `0.2` vs `0.5`
- Observación: Un Dropout de `0.2` mejora la precisión de test (`0.84369`) respecto a la línea base. El Dropout de `0.5` mejora ligeramente la línea base pero es inferior al de `0.2`.

| Nombre de Ejecución | Dropout | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|--------------------|---------|-----------|---------|----------|----------|----------------|-------------|
| r50_baseline       | 0.0     | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |
| r50_drop02         | 0.2     | 0.84885   | 0.9275  | 0.84369  | 0.83969  | 0.84807        | 0.82200     |
| r50_drop05         | 0.5     | 0.81578   | 0.9200  | 0.83413  | 0.82721  | 0.83971        | 0.82611     |

---

### Experimento 7: Comparación de Arquitecturas de Modelo

Comparación de diferentes arquitecturas ResNet bajo la misma configuración de entrenamiento.

- Comparación: ResNet18 vs ResNet34 vs ResNet50
- Observación: ResNet18 supera a ResNet50 en este dataset (`Test Acc = 0.84505` vs `0.82389`). El tamaño del dataset puede ser relativamente pequeño para ResNet50, lo que provoca sobreajuste o una optimización suboptimal.

| Nombre de Ejecución | Arquitectura | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|--------------------|--------------|-----------|---------|----------|----------|----------------|-------------|
| r18_baseline      | ResNet18     | 0.87580   | 0.9275  | 0.84505  | 0.83894  | 0.85220        | 0.83632     |
| r34_baseline      | ResNet34     | 0.85496   | 0.9325  | 0.82321  | 0.81843  | 0.82728        | 0.81657     |
| r50_baseline      | ResNet50     | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |

---

### Experimento 8: Escalamiento de Batch Size + LR

Aplicación de la Regla de Escalamiento Lineal: escalar la tasa de aprendizaje linealmente con el tamaño del lote (batch size).

- Comparación: Batch 32 (`LR = 0.01`) vs Batch 64 (`LR = 0.02`, Línea base) vs Batch 128 (`LR = 0.04`)
- Observación: La configuración `Batch 128 / LR 0.04` logra la mejor precisión de test (`0.85324`), la más alta entre todos los experimentos.

| Nombre de Ejecución | Batch / LR | Train Acc | Val Acc | Test Acc | Test F1  | Test Precision | Test Recall |
|--------------------|------------|-----------|---------|----------|----------|----------------|-------------|
| r50_b32_lr01       | B32 / 0.01 | 0.81134   | 0.9350  | 0.83003  | 0.82430  | 0.83846        | 0.82116     |
| r50_baseline       | B64 / 0.02 | 0.86413   | 0.9275  | 0.82389  | 0.81890  | 0.82830        | 0.81646     |
| r50_b128_lr04      | B128 / 0.04| 0.86885   | 0.9250  | 0.85324  | 0.84695  | 0.86186        | 0.84506     |

---
