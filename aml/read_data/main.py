import argparse
import os
import shrike
from shrike.compliant_logging.exceptions import prefix_stack_trace
from shrike.compliant_logging.constants import DataCategory
import logging
import pandas as pd
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="SystemLog:")
parser.add_argument("--log_level", default="INFO")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--output_dir", type=str, help="Path to output csv data")


@prefix_stack_trace(keep_message=True)
def main(args):
    
    shrike.compliant_logging.enable_compliant_logging(
        args.prefix,
        level=args.log_level,
        format="%(prefix)s%(levelname)s:%(name)s:%(message)s",
    )

    logger = logging.getLogger(__name__)

    logger.info("public info", category=DataCategory.PUBLIC)
    logger.info("Hello, world!", category=DataCategory.PUBLIC)

    lines = [
    f'Training data path: {args.training_data}',
    ]

    for line in lines:
        logger.info(line, category=DataCategory.PUBLIC)

    logger.info(os.listdir(args.training_data), category=DataCategory.PUBLIC)

    root = [args.training_data]
    all_files = set()
    while len(root) > 0:
        path = root.pop()
        for file in os.listdir(path):
            subpath = os.path.join(path, file)
            logger.info(subpath, category=DataCategory.PUBLIC)
            if os.path.isdir(subpath):
                root = [subpath] + root
            else:
                all_files.add(subpath)

    i = 0
    for file in all_files:
        df = pd.read_parquet(file)
        logger.info(df.head(), category=DataCategory.PUBLIC)
        i += 1
        if i > 10:
            break

    out_file = os.path.join(args.output_dir, "train.csv")
    all_lines = 0
    with open(out_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(["Subject", "UniqueBody"])

        for file in all_files:
            df = pd.read_parquet(file)
            for index, row in df.iterrows():
                if row["Subject"] and row["UniqueBody"] and len(str(row["Subject"])) > 5 and len(str(row["UniqueBody"])) > 5:
                    csvwriter.writerow([str(row["Subject"]), str(row["UniqueBody"])])
                    all_lines += 1

    logger.info(f"Total lines: {all_lines}", category=DataCategory.PUBLIC)

        # i = 0
        # with open(file, "r") as f:
        #     for line in f:
        #         logger.info(line, category=DataCategory.PUBLIC)
        #         i += 1
        #         if i > 10:
        #             break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)