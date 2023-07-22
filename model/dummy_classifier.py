from sklearn.dummy import DummyClassifier


class DummyModel:
    def __init__(self):
        self.model = DummyClassifier(strategy="most_frequent")

    def get_model(self):
        return self.model
