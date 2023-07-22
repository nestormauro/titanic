import argparse
import os
import subprocess

from config.settings import settings
from predict.predictor import Predict
from train.trainer import Train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML model for the Titanic dataset.")
    
    parser.add_argument(
        "option", choices=settings.PACKAGE_RUN_OPTIONS, help="Run options available"
    )
    parser.add_argument(
        "algorithm", choices=settings.AVAILABLE_ALGORITHMS, help="Algorithm name"
    )
    parser.add_argument(
        "--train_file_path",
        type=str,
        default=f"{os.getcwd()}/data/train.csv",
        help="Path to the train dataset",
    )
    parser.add_argument(
        "--test_file_path",
        type=str,
        default=f"{os.getcwd()}/data/test.csv",
        help="Path to the test dataset",
    )
    args = parser.parse_args()

    if args.option == "Train":
        train_obj = Train(args.algorithm)
        train_obj.train_model(args.train_file_path, args.test_file_path)
    elif args.option == "Predict":
        predict_obj = Predict(args.algorithm)
        predict_obj.set_predictions(args.train_file_path, args.test_file_path)
    elif args.option == "Test":
        subprocess.run(["python","-m","pytest"])