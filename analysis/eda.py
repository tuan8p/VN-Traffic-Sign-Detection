"""Offline data analysis for the detection set (boxes, not crops).

TODO [Team analysis + preprocessing offline] figures (write under analysis/figures/):
  01 class distribution of boxes
  02 box width/height
  03 box-center heatmap
  04 images-per-split
  05 tiny-box drop candidates
"""
from __future__ import annotations
import argparse

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Detection EDA (scaffold)")
    p.add_argument("--data-root", default=None)
    args = p.parse_args(argv)
    print("scaffold: EDA not implemented")
    print("data_root:", args.data_root or "(set DATA_ROOT / --data-root)")

if __name__ == "__main__":
    main()
