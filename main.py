from train.trainer import Train
import argparse

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="ML model for the Titanic dataset.")
    parser.add_argument("algorithm", choices=["Dummy", "RandomForest", "XGBoost"], help="Algorithm name")
    parser.add_argument("--train_file_path", type=str, default="../data/train.csv", help="Path to the train dataset")
    parser.add_argument("--test_file_path", type=str, default="../data/test.csv", help="Path to the test dataset")
    args = parser.parse_args()

    train_obj = Train(args.algorithm)
    train_obj.train_model(args.train_file_path, args.test_file_path)"""
    train_obj = Train("Dummy")
    train_obj.train_model("/home/nestor/workspace/python/ML/technical_assessments/titanic/data/train.csv", "/home/nestor/workspace/python/ML/technical_assessments/titanic/data/test.csv")