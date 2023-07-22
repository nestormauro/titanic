import os

import joblib

from etl.data_loader import DataLoader
from etl.feature_engineer import TitanicFeatureEngineer
from model.dummy_classifier import DummyModel
from model.gradient_boosting import GradientBoostingModel
from model.kfold_cross_validator import KFoldValidator
from model.random_forest import RandomForestModel


class Train:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def train_model(self, train_file_path, test_file_path):
        if self.algorithm == "Dummy":
            model = DummyModel().get_model()
        elif self.algorithm == "RandomForest":
            model = RandomForestModel().get_model()
        elif self.algorithm == "GradientBoosting":
            model = GradientBoostingModel().get_model()
        else:
            raise ValueError(f"Invalid algorithm name: {self.algorithm}")

        titanic_dl = DataLoader()
        titanic_dl.load_data(train_file_path, test_file_path)
        titanic_fe = TitanicFeatureEngineer(
            titanic_dl.get_train_df(), titanic_dl.get_test_df()
        )
        titanic_fe.feature_engineering()
        titanic_fe.feature_selection("RFE")
        X, y = titanic_fe.get_features_and_target()

        validator = KFoldValidator(model)
        (
            mean_roc_auc,
            std_roc_auc,
            mean_accuracy,
            std_accuracy,
        ) = validator.calculate_metrics(X, y)

        model.fit(X, y)

        joblib.dump(
            model,
            f"{os.getcwd()}/model/models/{str(type(model).__name__).lower()}.joblib",
        )

        print("------------------------")
        print("Training stage completed")
        print("------------------------")
        print("Mean ROC-AUC:", mean_roc_auc)
        print("Standard Deviation of ROC-AUC:", std_roc_auc)
        print("Mean Accuracy:", mean_accuracy)
        print("Standard Deviation of Accuracy:", std_accuracy)
