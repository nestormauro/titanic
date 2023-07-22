import os

from etl.data_loader import DataLoader
from etl.feature_engineer import TitanicFeatureEngineer
import joblib
from config.settings import settings

class Predict:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def set_predictions(self, train_file_path, test_file_path):
        if not str(self.algorithm).lower() in [x.lower() for x in settings.AVAILABLE_ALGORITHMS]:
            raise ValueError(f"Invalid algorithm name: {self.algorithm}")

        model_path = f"{os.getcwd()}/model/models/{str(self.algorithm).lower()}classifier.joblib"
        model_exists = os.path.isfile(model_path)

        if not model_exists:
            raise FileExistsError(f"The trained model was not found here: {model_path} please run the training process before")

        loaded_model = joblib.load(model_path)
        
        titanic_dl = DataLoader()
        titanic_dl.load_data(train_file_path, test_file_path)
        titanic_fe = TitanicFeatureEngineer(
            titanic_dl.get_train_df(), titanic_dl.get_test_df()
        )
        titanic_fe.feature_engineering()
        titanic_fe.feature_selection("RFE")
        test_df = titanic_fe.get_test_df()
        

        new_predictions = loaded_model.predict(test_df)
        print("Predictions for test data:", new_predictions)