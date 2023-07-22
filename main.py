import argparse
import os

from train.trainer import Train

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="ML model for the Titanic dataset.")
    parser.add_argument("algorithm", choices=["Dummy", "RandomForest", "GradientBoosting"], help="Algorithm name")
    parser.add_argument("--train_file_path", type=str, default="../data/train.csv", help="Path to the train dataset")
    parser.add_argument("--test_file_path", type=str, default="../data/test.csv", help="Path to the test dataset")
    args = parser.parse_args()

    train_obj = Train(args.algorithm)
    train_obj.train_model(args.train_file_path, args.test_file_path)"""
    train_obj = Train("GradientBoosting")
    train_obj.train_model(
        f"{os.getcwd()}/data/train.csv",
        f"{os.getcwd()}/data/test.csv",
    )
