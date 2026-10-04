from __future__ import annotations
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from vn_tsd.runtime.pack import zip_run_dir

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Zip a run directory the same way as classification")
    p.add_argument("--run-dir", required=True)
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit("missing run dir: " + str(run_dir))
    print(zip_run_dir(run_dir))

if __name__ == "__main__":
    main()
