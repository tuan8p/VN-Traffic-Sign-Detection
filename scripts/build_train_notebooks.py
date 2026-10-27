"""Regenerate unified 03/04 train notebooks."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def md_lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return [s]


def code_lines(s: str) -> list[str]:
    s = s.strip("\n") + "\n"
    return [line + ("\n" if not line.endswith("\n") else "") for line in s.splitlines(True)]


BOOTSTRAP = r'''import os, sys, subprocess
from pathlib import Path

REPO_URL = "https://github.com/tuan8p/VN-Traffic-Sign-Detection.git"
REPO_DIR = "VN-Traffic-Sign-Detection"
REPO_BRANCH = "{repo_branch}"  # faster-rcnn | yolo

base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
repo_path = base / REPO_DIR

def sh(cmd: str, check: bool = True) -> int:
    print(">>", cmd)
    return subprocess.run(cmd, shell=True, check=check).returncode

if (Path.cwd() / ".git").exists() and (Path.cwd() / "configs" / "shared.yaml").exists():
    repo_path = Path.cwd()
    print("Using the repo this notebook sits in:", repo_path)
elif not (repo_path / ".git").exists():
    sh(f'git clone --branch {{REPO_BRANCH}} {{REPO_URL}} "{{repo_path}}"')

git = f'git -C "{{repo_path}}"'
sh(f"{{git}} fetch origin {{REPO_BRANCH}}")
sh(f"{{git}} checkout -B {{REPO_BRANCH}} origin/{{REPO_BRANCH}}")

os.chdir(repo_path)
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))
print("cwd:", Path.cwd())
sh(f"{{git}} log --oneline -1")

sh('pip install -q -e ".[detect]"')
sh("pip install -q wandb")

import torch
import vn_tsd
from vn_tsd.config.resolve import resolve_config
print(
    f"Bootstrap OK. PyTorch {{torch.__version__}} | CUDA: {{torch.cuda.is_available()}} | GPUs: {{torch.cuda.device_count()}}"
)'''

DATA = r'''from pathlib import Path

# Dataset processed mặc định (Kaggle Add Data); thêm path khác phía trên nếu cần.
CANDIDATES = [
    "/kaggle/input/datasets/zero001206/vn-traffic-sign-detection-preprocessed-data/processed",
    "data/processed",
]

KAGGLE_YOLO26M = Path(
    "/kaggle/input/models/ultralytics/yolo26/pytorch/default/1/yolo26m.pt"
)
if KAGGLE_YOLO26M.is_file():
    print("[OK] YOLO26m weights:", KAGGLE_YOLO26M)
else:
    print("[!] Chưa thấy yolo26m.pt trên Kaggle — Add Model Ultralytics YOLO26 hoặc để Ultralytics tải khi train.")

def _ready(path: Path) -> bool:
    return (path / "data.yaml").exists() and (path / "images" / "train").is_dir()

PROCESSED_ROOT = next((c for c in CANDIDATES if c and _ready(Path(c))), None)
if PROCESSED_ROOT is None:
    print("[!] Chưa thấy cache processed (data.yaml + images/train).")
    print("    Chạy notebook 02 hoặc gắn dataset Kaggle vào CANDIDATES.")
    PROCESSED_ROOT = "data/processed"
else:
    print("[OK] Processed root:", PROCESSED_ROOT)'''

NOTEBOOK_SHARED = r'''# Override nhẹ trên configs/shared.yaml (phần còn lại lấy từ yaml khi resolve).
NOTEBOOK_SHARED = {
    "seed": 42,
    "train": {"batch_size": 24},
    "eval": {"batch_size": 32},
}
'''

RESOLVE = r'''from pathlib import Path
import copy
from vn_tsd.config.resolve import resolve_config
from vn_tsd.utils.io import save_yaml


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


ON_KAGGLE = Path("/kaggle/working").exists()

OVERRIDES = _deep_merge(NOTEBOOK_SHARED, NOTEBOOK_PIPELINE)
OVERRIDES.setdefault("data", {})["root"] = PROCESSED_ROOT
OVERRIDES.setdefault("train", {})["run_name"] = RUN_NAME

PIPELINE = NOTEBOOK_PIPELINE["pipeline"]
runtime_yaml = "configs/runtime/kaggle.yaml" if ON_KAGGLE else "configs/runtime/local.yaml"
cfg = resolve_config(
    pipeline_yaml=f"configs/pipelines/{PIPELINE}.yaml",
    shared_yaml="configs/shared.yaml",
    runtime_yaml=runtime_yaml,
    overrides=OVERRIDES,
)

OVERRIDES_PATH = Path("outputs") / "_notebook" / f"overrides_{PIPELINE}.yaml"
OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
save_yaml(OVERRIDES, OVERRIDES_PATH)

train = cfg.get("train", {})
print("=== Resolved configuration ===")
print("Run name        :", RUN_NAME)
print("Pipeline        :", cfg.get("pipeline"))
print("Epochs          :", train.get("epochs"))
print("Primary metric  :", train.get("primary_metric"))
print("Batch total     :", train.get("batch_size"))
print("Eval batch total:", cfg.get("eval", {}).get("batch_size"))
print("AMP             :", train.get("amp"))
print("W&B required    :", train.get("require_wandb"))
print("Online          :", cfg.get("online_preprocess", {}).get("enabled"))
print("Auto eval test  :", cfg.get("eval", {}).get("auto_eval_test"))
print("Data root       :", cfg.get("data", {}).get("root"))
print("Overrides file  :", OVERRIDES_PATH)'''

WANDB = r'''import os
from pathlib import Path

base_url = (cfg.get("project") or {}).get("wandb_base_url") or "https://forge.coreweave.com"
os.environ["WANDB_BASE_URL"] = base_url

if cfg.get("train", {}).get("require_wandb", True):
    if Path("/kaggle/working").exists():
        from kaggle_secrets import UserSecretsClient
        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY").strip()
        print("W&B key loaded from Kaggle Secrets.")
    elif not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY in the environment for local runs.")
    os.environ["WANDB_MODE"] = "online"
    print("Secret ready; rank 0 starts W&B inside tools.run_train.")
else:
    os.environ["WANDB_MODE"] = "disabled"
    print("W&B disabled for this run (train.require_wandb=False).")'''

YOLO_COPY = r'''from pathlib import Path
import shutil
import uuid
from vn_tsd.utils.io import load_yaml, save_yaml

source_root = Path(PROCESSED_ROOT).resolve()
if not (source_root / "data.yaml").is_file():
    raise FileNotFoundError(source_root / "data.yaml")
for split in ("train", "val", "test"):
    for kind in ("images", "labels"):
        if not (source_root / kind / split).is_dir():
            raise FileNotFoundError(source_root / kind / split)

working = Path("/kaggle/working")
size = sum(p.stat().st_size for p in source_root.rglob("*") if p.is_file())
if shutil.disk_usage(working).free < size * 1.2:
    raise RuntimeError("Not enough space to copy dataset to Working")

fixed_root = working / f"yolo_dataset_{uuid.uuid4().hex[:8]}"
shutil.copytree(source_root, fixed_root, ignore=shutil.ignore_patterns("*.cache"))
data_cfg = load_yaml(fixed_root / "data.yaml")
data_cfg.update(path=str(fixed_root), train="images/train", val="images/val", test="images/test")
save_yaml(data_cfg, fixed_root / "data.yaml")
PROCESSED_ROOT = str(fixed_root)
print("Writable dataset copy:", PROCESSED_ROOT)'''

FRCNN_DATA_PREP = r'''# Faster R-CNN đọc cache read-only; không cần copy sang Working.
from pathlib import Path
print("Using PROCESSED_ROOT:", Path(PROCESSED_ROOT).resolve())'''

LAUNCH = r'''import os
import sys
import subprocess
from pathlib import Path

import torch
from vn_tsd.runtime.run_dir import make_run_dir
from vn_tsd.utils.io import save_yaml

ON_KAGGLE = Path("/kaggle/working").exists()
RUNTIME = "configs/runtime/kaggle.yaml" if ON_KAGGLE else "configs/runtime/local.yaml"

if ON_KAGGLE:
    assert torch.cuda.device_count() >= 2, "Chọn accelerator 2× GPU trên Kaggle."
    assert cfg["train"]["batch_size"] % 2 == 0, "Global batch phải chia hết cho 2 GPU."

# Output luôn dưới repo clone: <repo>/outputs/<pipeline>/<run_id>/
OUTPUTS_ROOT = Path.cwd() / "outputs"
RUN_DIR = make_run_dir(OUTPUTS_ROOT, PIPELINE, RUN_NAME)

OVERRIDES["data"]["root"] = PROCESSED_ROOT
OVERRIDES["train"]["run_name"] = RUN_NAME
launch_overrides = RUN_DIR / "launch_overrides.yaml"
save_yaml(OVERRIDES, launch_overrides)

cmd = [
    "--pipeline", PIPELINE,
    "--runtime", RUNTIME,
    "--run-name", RUN_NAME,
    "--overrides", str(launch_overrides),
    "--data-root", PROCESSED_ROOT,
    "--online" if cfg.get("online_preprocess", {}).get("enabled") else "--no-online",
]
cmd.extend(["--run-dir", str(RUN_DIR)])

if ON_KAGGLE:
    train_cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "-m", "tools.run_train", *cmd]
else:
    train_cmd = [sys.executable, "-m", "tools.run_train", *cmd]

print(">> ", " ".join(train_cmd))
print("run_dir (before train):", RUN_DIR)
proc = subprocess.run(train_cmd, check=False)
if proc.returncode != 0:
    import json

    print("\n=== torchrun failed (exit", proc.returncode, ") ===")
    print("Scroll up in this cell for rank-0 traceback (CUDA OOM, W&B, ValueError, …).")
    detail_parts: list[str] = []
    failure = RUN_DIR / "failure.json"
    if failure.is_file():
        text = failure.read_text(encoding="utf-8")
        print("\n--- failure.json ---")
        print(text)
        detail_parts.append(text.strip())
    metrics = RUN_DIR / "metrics.json"
    if metrics.is_file():
        try:
            doc = json.loads(metrics.read_text(encoding="utf-8"))
            if doc.get("status") == "failed":
                block = json.dumps(doc, indent=2)
                print("\n--- metrics.json (failed) ---")
                print(block)
                if not detail_parts:
                    detail_parts.append(block)
        except json.JSONDecodeError:
            pass
    msg = detail_parts[0] if detail_parts else (
        f"Training exited with code {proc.returncode}. See traceback above in this cell output."
    )
    raise RuntimeError(msg)

print("run_dir:", RUN_DIR)
print("wandb:", RUN_DIR / "wandb_run.json")
print("zip:", RUN_DIR / "run.zip")
'''

FRCNN_SPECIFIC = r'''# Mirror configs/pipelines/faster_rcnn.yaml
NOTEBOOK_PIPELINE = {
    "pipeline": "faster_rcnn",
    "model": {
        "head": "custom_mlp",
        "head_hidden_dim": 512,
        "head_dropout": 0.2,
        "weights": "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
        "trainable_backbone_layers": 5,
        "frozen_bn_eps": 0.0,
        "score_thresh": 0.001,
        "nms_thresh": 0.5,
        "detections_per_img": 12,
    },
    "train": {
        "epochs": 100,
        "lr": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "warmup_start_factor": 0.001,
        "min_lr": 0.000001,
    },
    "online_preprocess": {
        "enabled": False,
        "letterbox": True,
        "normalize": True,
        "hflip": False,
        "color_jitter": False,
        "mosaic": False,
        "mixup": False,
        "jitter": 0.1,
        "mosaic_prob": 1.0,
        "mixup_prob": 0.1,
        "mixup_alpha": 32.0,
        "color_jitter_params": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.02,
        },
    },
}

import uuid

_suffix = "_online" if NOTEBOOK_PIPELINE["online_preprocess"]["enabled"] else ""
_ep = NOTEBOOK_PIPELINE["train"]["epochs"]
_bs = NOTEBOOK_SHARED["train"]["batch_size"]
RUN_NAME = (
    f"faster_rcnn_s{NOTEBOOK_SHARED['seed']}_ep{_ep}_bs{_bs}{_suffix}_{uuid.uuid4().hex[:8]}"
)
'''

YOLO_SPECIFIC = r'''# Mirror configs/pipelines/yolo26m.yaml (weights/imgsz/family lấy từ yaml mặc định)
NOTEBOOK_PIPELINE = {
    "pipeline": "yolo26m",
    "model": {
        "score_thresh": 0.001,
        "nms_thresh": 0.5,
        "detections_per_img": 12,
    },
    "train": {
        "epochs": 100,
        "lr0": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "lrf": 0.01,
        "warmup_epochs": 5.0,
        "cos_lr": True,
        "close_mosaic": 10,
    },
    "online_preprocess": {
        "enabled": False,
        "letterbox": True,
        "normalize": True,
        "hflip": False,
        "color_jitter": False,
        "mosaic": True,
        "mixup": False,
        "jitter": 0.1,
        "mosaic_prob": 1.0,
        "mixup_prob": 0.1,
        "mixup_alpha": 32.0,
        "color_jitter_params": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.02,
        },
    },
}

import uuid

_suffix = "_online" if NOTEBOOK_PIPELINE["online_preprocess"]["enabled"] else ""
_ep = NOTEBOOK_PIPELINE["train"]["epochs"]
_bs = NOTEBOOK_SHARED["train"]["batch_size"]
RUN_NAME = (
    f"yolo26m_s{NOTEBOOK_SHARED['seed']}_ep{_ep}_bs{_bs}{_suffix}_{uuid.uuid4().hex[:8]}"
)
'''


def build_nb(meta: dict) -> dict:
    cells: list[dict] = []
    cells.append({"cell_type": "markdown", "metadata": {}, "source": md_lines(meta["title_md"])})
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": md_lines(
                "## 0) Bootstrap (Kaggle / Colab)\n\nClone hoặc pull nhánh, rồi cài dependency. Trên Kaggle bật Internet: On.\n"
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines(BOOTSTRAP.format(repo_branch=meta["repo_branch"])),
        }
    )
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": md_lines(
                "## 1) Đường dẫn data/processed\n\nCache offline dùng chung (letterbox 640, nhãn YOLO). Chỉ pretrain weights là tài sản ngoài repo.\n"
            ),
        }
    )
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_lines(DATA)})
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": md_lines(
                "## 2) Override dùng chung\n\n"
                "Chỉ seed / batch train / batch eval; còn lại từ `configs/shared.yaml` + `configs/runtime/*.yaml`.\n"
            ),
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines(meta["notebook_shared"]),
        }
    )
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": md_lines("## 3) Hyperparameters riêng pipeline\n\n") + md_lines(meta["specific_md"]),
        }
    )
    cells.append(
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_lines(meta["specific_code"])}
    )
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": md_lines(
                "## 4) Resolve config\n\n"
                "Merge notebook dicts → `OVERRIDES`, rồi `resolve_config` (yaml repo làm baseline + validate schema).\n"
            ),
        }
    )
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_lines(RESOLVE)})
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": md_lines(
                "## 5) W&B Secrets\n\nNotebook **không** gọi `wandb.init()`; rank 0 trong `tools.run_train` mở run training.\n"
            ),
        }
    )
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_lines(WANDB)})
    cells.append({"cell_type": "markdown", "metadata": {}, "source": md_lines(meta["data_prep_md"])})
    cells.append(
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_lines(meta["data_prep_code"])}
    )
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": md_lines(
                "## 7) Launch train\n\nKaggle: `torchrun` 2 GPU. Output: `outputs/<pipeline>/` trong repo. YOLO: test 1 GPU tự chạy sau DDP (rank 0). Best = val mAP50-95.\n"
            ),
        }
    )
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_lines(LAUNCH)})
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def main() -> None:
    (REPO / "notebooks" / "03_train_faster_rcnn.ipynb").write_text(
        json.dumps(
            build_nb(
                {
                    "title_md": "# 03 — Train Faster R-CNN ResNet-50\n\nTorchvision Faster R-CNN ResNet-50, 52 lớp. Best = val mAP50-95 (COCOeval). Test một lần sau best.\n",
                    "repo_branch": "faster-rcnn",
                    "pipeline": "faster_rcnn",
                    "notebook_shared": NOTEBOOK_SHARED,
                    "specific_md": "Dict `NOTEBOOK_PIPELINE` mirror `configs/pipelines/faster_rcnn.yaml`.\n",
                    "specific_code": FRCNN_SPECIFIC,
                    "data_prep_md": "## 6) Dataset\n\nFRCNN đọc trực tiếp Input read-only; **không** copy sang Working.\n",
                    "data_prep_code": FRCNN_DATA_PREP,
                }
            ),
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    (REPO / "notebooks" / "04_train_yolo26m.ipynb").write_text(
        json.dumps(
            build_nb(
                {
                    "title_md": "# 04 — Train YOLO26m\n\nUltralytics YOLO26m, 52 lớp. Best = val mAP50-95 (COCOeval). Test (nếu bật) qua `tools.run_eval` sau DDP.\n",
                    "repo_branch": "yolo",
                    "pipeline": "yolo26m",
                    "notebook_shared": NOTEBOOK_SHARED,
                    "specific_md": "Dict `NOTEBOOK_PIPELINE` mirror `configs/pipelines/yolo26m.yaml`.\n",
                    "specific_code": YOLO_SPECIFIC,
                    "data_prep_md": "## 6) Dataset (YOLO)\n\nCopy **thật** sang `/kaggle/working` (Ultralytics ghi label cache; symlink Input không đủ).\n",
                    "data_prep_code": YOLO_COPY,
                }
            ),
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print("Wrote notebooks/03_train_faster_rcnn.ipynb and 04_train_yolo26m.ipynb")


if __name__ == "__main__":
    main()
