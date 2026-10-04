from __future__ import annotations
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from vn_tsd.config.resolve import resolve_config

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Offline preprocess (shared, detection boxes)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--shared", default="configs/shared.yaml")
    args = p.parse_args(argv)
    cfg = resolve_config(shared_yaml=args.shared)
    print("scaffold: offline preprocess not implemented yet")
    print("data_root:", args.data_root)
    print("image_size:", cfg.get("preprocess", {}).get("image_size"))
    # TODO [Team analysis + preprocessing offline]: vn_tsd.data.preprocess_offline.run_offline(cfg, args.data_root)

if __name__ == "__main__":
    main()
