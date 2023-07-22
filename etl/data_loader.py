import pandas as pd


class DataLoader:
    def __init__(self):
        self.train_df = None
        self.test_df = None

    def load_data(self, train_file_path, test_file_path):
        self.train_df = pd.read_csv(train_file_path, index_col="PassengerId")
        self.test_df = pd.read_csv(test_file_path, index_col="PassengerId")

    def get_train_df(self):
        return self.train_df

    def get_test_df(self):
        return self.test_df
