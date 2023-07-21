from etl.feature_engineer import TitanicFeatureEngineer
from model.dummy_classifier import DummyModel
from model.kfold_cross_validator import KFoldValidator

class Train:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def train_model(self, train_file_path, test_file_path):
        if self.algorithm == "Dummy":
            model = DummyModel().get_model()
        elif self.algorithm == "RandomForest":
            #model = RandomForestModel().get_model()
            pass
        elif self.algorithm == "XGBoost":
            #model = XGBoostModel().get_model()
            pass
        else:
            raise ValueError(f"Invalid algorithm name: {self.algorithm}")

        titanic_fe = TitanicFeatureEngineer(train_file_path, test_file_path)
        titanic_fe.load_data()
        titanic_fe.feature_engineering()
        titanic_fe.feature_selection("RFE")
        train_df = titanic_fe.get_train_df()
        test_df = titanic_fe.get_test_df()

        validator = KFoldValidator(model)
        validator.plot_confusion_matrix(train_df, test_df)
        mean_roc_auc, std_roc_auc, mean_accuracy, std_accuracy = validator.calculate_metrics(train_df, test_df)

        print("Mean ROC-AUC:", mean_roc_auc)
        print("Standard Deviation of ROC-AUC:", std_roc_auc)
        print("Mean Accuracy:", mean_accuracy)
        print("Standard Deviation of Accuracy:", std_accuracy)