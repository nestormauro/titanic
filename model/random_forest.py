from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(random_state)

    def get_model(self):
        return self.model
