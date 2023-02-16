import argparse
import os
import shrike
from shrike.compliant_logging.exceptions import prefix_stack_trace
from shrike.compliant_logging.constants import DataCategory
import logging
import datasets
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

    data_path = os.path.join(args.training_data, "train.csv")
    dataset = datasets.load_dataset('csv', data_files={'train': data_path})

    dataset2 = dataset['train'].train_test_split(test_size=0.3)
    train_dataset = dataset2['train']

    dataset3 = dataset2['test'].train_test_split(test_size=0.5)
    val_dataset = dataset3['train']
    test_dataset = dataset3['test']

    train_dataset.to_csv(os.path.join(args.output_dir, "train.csv"))
    val_dataset.to_csv(os.path.join(args.output_dir, "val.csv"))
    test_dataset.to_csv(os.path.join(args.output_dir, "test.csv"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)