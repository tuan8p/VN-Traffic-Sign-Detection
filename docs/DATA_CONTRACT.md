# Data contract

Source is a YOLO detection set (images + labels), same family as VNTS (52 classes, `configs/classes.csv`). This repo does **not** crop boxes into classification samples.

## Expected raw layout

```
DATA_ROOT/
  images/
  labels/          # YOLO txt: class cx cy w h (normalized)
  split_dataset/   # optional provided splits
```

## Processed layout (offline)

TODO [Team analysis + preprocessing offline]: hiện thực layout này trong vn_tsd/data/preprocess_offline.py.

```
data/processed/
  images/{train,val,test}/
  labels/{train,val,test}/
  data.yaml
  meta.json
```

## Rules

- Horizontal flip stays off by default (mirror-pair sign classes).
- Primary metric is mAP@0.50:0.95. Also report mAP@0.50, mAP@0.75, and AR@100. Same keys as configs/shared.yaml: mAP50-95, mAP50, mAP75, AR100.
- Both pipelines read the same processed split. Online aug is a tuning switch, not part of the cached set.

Offline preprocessing là bản finetune chung. Nó gồm letterbox và augmentation phù hợp (không hflip). Không lưu thành .npy crop. Cả hai pipeline đọc cùng cây này.
