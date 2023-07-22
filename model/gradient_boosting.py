from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingModel:
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def get_model(self):
        return self.model
