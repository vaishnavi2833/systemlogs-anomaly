import os
import time
import numpy as np
import pandas as pd
from process.sliding_window_processor import collect_event_ids, FeatureExtractor

def process_logs(load_data_dir: str, save_base_dir: str, version_suffix: str = "_v5", use_tf_idf: bool = True,structured_log_filename = str):
    """
    Loads parsed logs and processes them into CNN-compatible input data using a sliding window.
    No splitting into train/test. Entire input is used for processing.

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
    df_logs = pd.read_csv(os.path.join(load_data_dir, structured_log_filename))
    y = pd.read_csv(os.path.join(load_data_dir, "anomaly_label.csv"))

    print("Extracting event sequences...")
    re_pat = r"(blk_-?\d+)"
    col_names = ["BlockId", "EventSequence"]
    events = collect_event_ids(df_logs, re_pat, col_names)

    print("Merging with anomaly labels...")
    events = events.merge(y, on="BlockId")

    events_values = events["EventSequence"].values

    print("Calculating dynamic window size...")
    min_seq_len = min(len(seq) for seq in events_values)
    window_size = min(16, min_seq_len)

    print("Applying Feature Extraction...")
    fe = FeatureExtractor()
    subblocks = fe.fit_transform(
        events_values,
        term_weighting="tf-idf" if use_tf_idf else None,
        length_percentile=95,
        window_size=window_size,
    )

    print("Saving processed data...")
    # Save labels
    events[["BlockId", "Label"]].to_csv(os.path.join(save_dir, f"y{data_version}.csv"), index=False)
    
    # Save numpy array
    np.save(os.path.join(save_dir, f"x{data_version}.npy"), subblocks)

    print(f"✅ Done! Time taken: {round(time.time() - start, 2)} seconds")
