from config.settings import settings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score


class KFoldValidator:
    def __init__(self, algorithm, num_folds=5):
        self.algorithm = algorithm
        self.num_folds = num_folds
        self.kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    def calculate_metrics(self, train_df):
        X = train_df.drop(settings.TARGET, axis=1)
        y = train_df[settings.TARGET]

        roc_auc_scores = cross_val_score(
            self.algorithm, X, y, scoring="roc_auc", cv=self.kfold
        )
        accuracy_scores = cross_val_score(
            self.algorithm, X, y, scoring="accuracy", cv=self.kfold
        )

        mean_roc_auc = roc_auc_scores.mean()
        std_roc_auc = roc_auc_scores.std()
        mean_accuracy = accuracy_scores.mean()
        std_accuracy = accuracy_scores.std()

        self._plot_confusion_matrix(train_df)

        return mean_roc_auc, std_roc_auc, mean_accuracy, std_accuracy

    def _plot_confusion_matrix(self, train_df):
        X = train_df.drop(settings.TARGET, axis=1)
        y = train_df[settings.TARGET]
        y_pred = cross_val_predict(self.algorithm, X, y, cv=self.kfold)
        cm = confusion_matrix(y, y_pred)
        algorithm_name = type(self.algorithm).__name__

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig(f"confusion_matrix_{algorithm_name}.png")
