from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.dummy import DummyClassifier

class DummyModel:
    def __init__(self):
        self.model = DummyClassifier(strategy='most_frequent')

    def get_model(self):
        return self.model