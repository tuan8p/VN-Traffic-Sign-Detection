from __future__ import annotations
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a detection run")
    p.add_argument("--pipeline", required=True, choices=["faster_rcnn", "yolo26m"])
    p.add_argument("--run-dir", required=True)
    args = p.parse_args(argv)
    print("TODO [Team pipeline YOLO26m + preprocessing online] / [Team pipeline Faster R-CNN + preprocessing online]: eval not implemented. Fill only your pipeline branch.")
    print("pipeline:", args.pipeline)
    print("run_dir:", Path(args.run_dir))
    # TODO [Team pipeline YOLO26m + preprocessing online]: if pipeline is yolo26m, load the Ultralytics checkpoint and report mAP.
    # TODO [Team pipeline Faster R-CNN + preprocessing online]: if pipeline is faster_rcnn, load the torchvision checkpoint and report mAP.

if __name__ == "__main__":
    main()
