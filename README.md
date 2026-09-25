# Vehicle Detection

Custom Faster R-CNN (ResNet + FPN) and a custom YOLO detector trained on the same single-class vehicle dataset.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data is included under `data/dataset/`.

## Usage

```bash
python -m faster_rcnn.train
python -m faster_rcnn.demo

python -m yolo.train
python -m yolo.demo
```

Training selects DataLoader settings with `--runtime {auto,local,colab}` (default: `auto`).

| Runtime | YOLO batch / workers | Faster R-CNN batch / workers |
| ------- | -------------------- | ---------------------------- |
| `local` | 8 / 0                | 4 / 0                        |
| `colab` | 32 / 2               | 12 / 2                       |

Optional flags: `--epochs`, `--img-size`, `--max-batches`.

## Colab

```python
from google.colab import drive
drive.mount("/content/drive")

import os
os.environ["PROJECT_ROOT"] = "/content/drive/MyDrive/vehicle detection"
%cd {os.environ["PROJECT_ROOT"]}
!pip install -q -r requirements.txt

!python -m yolo.train --runtime colab
!python -m yolo.demo
```

Set `PROJECT_ROOT` to the project directory if it is not discovered automatically.
