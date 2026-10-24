"""analysis package – EDA for the VN-Traffic-Sign-Detection project."""
from analysis.image_utils import crop_active_region, calculate_dhash, calculate_brightness_sharpness
from analysis.class_distribution import plot_class_distribution
from analysis.clustering import cluster_images_dhash, plot_dhash_clusters
from analysis.leakage import check_data_leakage, plot_data_leakage, find_leaked_test_stems

__all__ = [
    "crop_active_region",
    "calculate_dhash",
    "calculate_brightness_sharpness",
    "plot_class_distribution",
    "cluster_images_dhash",
    "plot_dhash_clusters",
    "check_data_leakage",
    "plot_data_leakage",
    "find_leaked_test_stems",
    "split_by_dhash_clusters",
]


def __getattr__(name: str):
    if name == "split_by_dhash_clusters":
        from vn_tsd.data.split import split_by_dhash_clusters
        return split_by_dhash_clusters
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
