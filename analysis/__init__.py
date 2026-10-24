from analysis.helpers.image_utils import crop_active_region, calculate_dhash, calculate_brightness_sharpness
from analysis.helpers.class_distribution import plot_class_distribution
from analysis.helpers.clustering import cluster_images_dhash, plot_dhash_clusters
from analysis.helpers.leakage import check_data_leakage, plot_data_leakage, find_leaked_test_stems

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
    "analyze_raw_dataset",
    "analyze_processed_dataset",
    "split_by_dhash_clusters",
]


def __getattr__(name: str):
    if name == "split_by_dhash_clusters":
        from vn_tsd.data.helpers.split import split_by_dhash_clusters
        return split_by_dhash_clusters
    if name == "analyze_raw_dataset":
        from analysis.eda_raw import analyze_raw_dataset
        return analyze_raw_dataset
    if name == "analyze_processed_dataset":
        from analysis.eda_processed import analyze_processed_dataset
        return analyze_processed_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
