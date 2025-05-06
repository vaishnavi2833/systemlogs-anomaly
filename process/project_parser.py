import os
import pandas as pd
from process.logparser import Drain
from process.project_processor import process_logs  

def parse_logs(input_path: str, output_dir: str, save_processed_dir: str):
    """
    Parses the raw log using the Drain parser and processes it into structured + model input.
    """
    
    # log_format = "<Date> <Time> <Pid> <Level> <Component>: <Content>"

    # regex = [
    #     r"blk_(|-)[0-9]+",  # Block ID
    #     r"(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)",  # IP Address
    #     r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$",  # Numbers
    # ]

    # st = 0.5
    # depth = 4

    # os.makedirs(output_dir, exist_ok=True)

    # log_filename = os.path.basename(input_path)
    # log_file_train = os.path.join(os.path.dirname(input_path), "HDFS_train.log")
    # parsed_csv_path = os.path.join(output_dir, log_filename + "_structured.csv")

    # with open(input_path, "r") as file:
    #     lines = file.readlines()
    # totaln = len(lines)
    # print(f"There are a total of {totaln} lines")

    # train_idx = int(totaln * 0.8)
    # train_lines = lines[:train_idx]
    # test_lines = lines[train_idx:]

    # with open(log_file_train, "w") as f:
    #     f.writelines(train_lines)

    # original_lines_path = os.path.join(save_processed_dir, "original_lines.txt")
    # with open(original_lines_path, "w") as f:
    #     f.writelines(test_lines)

    # parser = Drain.LogParser(
    #     log_format, indir=os.path.dirname(input_path), outdir=output_dir,
    #     depth=depth, st=st, rex=regex
    # )

    # parser.parse(log_filename)               # Parse full log
    # parser.parse("HDFS_train.log")             # Parse train log

    # all_parsed_path = os.path.join(output_dir, log_filename + "_structured.csv")
    # all_parsed_df = pd.read_csv(all_parsed_path)
    # test_parsed_df = all_parsed_df.iloc[train_idx:]
    # test_parsed_df.to_csv(os.path.join(output_dir, "HDFS_test.log_structured.csv"), index=False)

    print("✅ Parsing complete.")
    # print("\n🔄 Starting Processing into model inputs...")
    process_logs(load_data_dir=output_dir, save_base_dir=save_processed_dir)
    print("✅ Full preprocessing pipeline completed!")
