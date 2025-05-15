import os
import pandas as pd
from process.logparser import Drain
from process.project_processor import process_logs  

def parse_logs(input_path: str, output_dir: str, save_processed_dir: str):
    """
    Parses the raw log using the Drain parser and processes it into structured + model input.
    No splitting into train/test.
    """
    
    log_format = "<Date> <Time> <Pid> <Level> <Component>: <Content>"

    regex = [
        r"blk_(|-)[0-9]+",  # Block ID
        r"(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)",  # IP Address
        r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$",  # Numbers
    ]

    st = 0.5
    depth = 4

    os.makedirs(output_dir, exist_ok=True)

    log_filename = os.path.basename(input_path)
    parsed_csv_path = os.path.join(output_dir, log_filename + "_structured.csv")
    output_path = (log_filename +  "_structured.csv")


    with open(input_path, "r") as file:
        lines = file.readlines()
    print(f"There are a total of {len(lines)} lines")

    # Save all lines as original input (no split)
    original_lines_path = os.path.join(save_processed_dir, "original_lines.txt")
    with open(original_lines_path, "w") as f:
        f.writelines(lines)

    parser = Drain.LogParser(
        log_format, indir=os.path.dirname(input_path), outdir=output_dir,
        depth=depth, st=st, rex=regex
    )

    parser.parse(log_filename)  

    print("✅ Parsing complete.")
    print("\n🔄 Starting Processing into model inputs...")
    process_logs(load_data_dir=output_dir, save_base_dir=save_processed_dir,structured_log_filename = output_path)
    print("✅ Full preprocessing pipeline completed!")
