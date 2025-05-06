import os
import time
import numpy as np
import pandas as pd
from process.sliding_window_processor import collect_event_ids, FeatureExtractor

def process_logs(load_data_dir: str, save_base_dir: str, version_suffix: str = "_v5", use_tf_idf: bool = True):
    """
    Loads parsed logs and processes them into CNN-compatible input data using a sliding window.

    Args:
        load_data_dir (str): Directory containing parsed log files and anomaly labels.
        save_base_dir (str): Base directory to save the processed data.
        version_suffix (str): Optional suffix for data versioning (default: "_v5").
        use_tf_idf (bool): Whether to apply TF-IDF weighting (default: True).
    """
    start = time.time()

    # Build versioned folder path
    data_version = f"{'_tf-idf' if use_tf_idf else ''}{version_suffix}"
    save_dir = os.path.join(save_base_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("Loading CSV files...")
    x_train = pd.read_csv(os.path.join(load_data_dir, "HDFS_train.log_structured.csv"))
    x_test = pd.read_csv(os.path.join(load_data_dir, "HDFS_test.log_structured.csv"))
    y = pd.read_csv(os.path.join(load_data_dir, "anomaly_label.csv"))

    print("Extracting event sequences...")
    re_pat = r"(blk_-?\d+)"
    col_names = ["BlockId", "EventSequence"]
    events_train = collect_event_ids(x_train, re_pat, col_names)
    events_test = collect_event_ids(x_test, re_pat, col_names)

    print("Merging with anomaly labels...")
    events_train = events_train.merge(y, on="BlockId")
    events_test = events_test.merge(y, on="BlockId")

    print("Filtering overlapping blocks...")
    overlapping_blocks = np.intersect1d(events_train["BlockId"], events_test["BlockId"])
    events_train = events_train[~events_train["BlockId"].isin(overlapping_blocks)]
    events_test = events_test[~events_test["BlockId"].isin(overlapping_blocks)]

    events_train_values = events_train["EventSequence"].values
    events_test_values = events_test["EventSequence"].values

    print("Calculating dynamic window size...")
    min_seq_len = min(len(seq) for seq in events_train_values)
    window_size = min(16, min_seq_len) 

    print("Applying Feature Extraction...")
    fe = FeatureExtractor()
    subblocks_train = fe.fit_transform(
        events_train_values,
        term_weighting="tf-idf" if use_tf_idf else None,
        length_percentile=95,
        window_size=window_size,
    )
    subblocks_test = fe.transform(events_test_values)

    print("Saving processed data...")
    # Save labels
    y_train = events_train[["BlockId", "Label"]]
    y_test = events_test[["BlockId", "Label"]]
    y_train.to_csv(os.path.join(save_dir, f"y_train{data_version}.csv"), index=False)
    y_test.to_csv(os.path.join(save_dir, f"y_test{data_version}.csv"), index=False)

    # Save numpy arrays
    np.save(os.path.join(save_dir, f"x_train{data_version}.npy"), subblocks_train)
    np.save(os.path.join(save_dir, f"x_test{data_version}.npy"), subblocks_test)

    print(f"✅ Done! Time taken: {round(time.time() - start, 2)} seconds")
