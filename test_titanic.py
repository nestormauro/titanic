import os
import pytest
from etl.data_loader import DataLoader
from etl.feature_engineer import TitanicFeatureEngineer
from model.dummy_classifier import DummyModel
from model.gradient_boosting import GradientBoostingModel
from model.random_forest import RandomForestModel
from model.kfold_cross_validator import KFoldValidator


class TestTitanic:
    def setup_method(self, method):
        self.train_file_path = f"{os.getcwd()}/test_data/train.csv"
        self.test_file_path = f"{os.getcwd()}/test_data/test.csv"

    def test_data_loader(self):
        dataloader = DataLoader()
        dataloader.load_data(self.train_file_path, self.test_file_path)

        train_df = dataloader.get_train_df()
        test_df = dataloader.get_test_df()

        assert train_df is not None
        assert test_df is not None
        assert len(list(train_df.columns)) == 11
        assert len(list(test_df.columns)) == 10

    def test_feature_engineering(self):
        dataloader = DataLoader()
        dataloader.load_data(self.train_file_path, self.test_file_path)

        train_df = dataloader.get_train_df()
        test_df = dataloader.get_test_df()

        feature_engineer = TitanicFeatureEngineer(train_df, test_df)
        feature_engineer.feature_engineering()
        feature_engineer.feature_selection("RFE")
        X, y = feature_engineer.get_features_and_target()


        assert len(list(feature_engineer.get_train_df().columns)) == 21
        assert len(list(feature_engineer.get_test_df().columns)) == 20

    def test_dummy_model(self):
        model = DummyModel().get_model()
        assert str(type(model).__name__).lower() == 'dummyclassifier'

    def test_random_forest_model(self):
        model = RandomForestModel().get_model()
        assert str(type(model).__name__).lower() == 'randomforestclassifier'

    def test_gradient_boosting_model(self):
        model = GradientBoostingModel().get_model()
        assert str(type(model).__name__).lower() == 'gradientboostingclassifier'

    def test_kfold_validator(self):
        dataloader = DataLoader()
        dataloader.load_data(self.train_file_path, self.test_file_path)

        train_df = dataloader.get_train_df()
        test_df = dataloader.get_test_df()

        feature_engineer = TitanicFeatureEngineer(train_df, test_df)
        feature_engineer.feature_engineering()
        feature_engineer.feature_selection("RFE")
        X, y = feature_engineer.get_features_and_target()

        model = DummyModel().get_model()
        validator = KFoldValidator(model)
        mean_roc_auc, std_roc_auc, mean_accuracy, std_accuracy = validator.calculate_metrics(X, y)

        assert mean_roc_auc >= 0.0
        assert std_roc_auc >= 0.0
        assert mean_accuracy >= 0.0
        assert std_accuracy >= 0.0

    def test_train_random_forest_model(self):
        dataloader = DataLoader()
        dataloader.load_data(self.train_file_path, self.test_file_path)

        train_df = dataloader.get_train_df()
        test_df = dataloader.get_test_df()

        feature_engineer = TitanicFeatureEngineer(train_df, test_df)
        feature_engineer.feature_engineering()
        feature_engineer.feature_selection("RFE")
        X, y = feature_engineer.get_features_and_target()

        model = DummyModel().get_model()
        model.fit(X, y)

        assert model.score(X, y) > 0