from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET = "Survived"
UNNECESSARY_FEATURES = ["SibSp", "Parch", "Name", "Title", "title", "Ticket", "Cabin"]
NON_NUMERIC_FEATURES = [
    "Embarked",
    "Sex",
    "CabinLetter",
    "SocialStatus",
    "FamilySize",
    "Age",
    "Fare",
]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "CabinLetter"]
CONTINUOUS_FEATURES = ["FamilySize", "Age", "Fare"]


class TitanicFeatureEngineer:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.train_df = None
        self.test_df = None

    def load_data(self):
        # Load the Titanic dataset
        self.train_df = pd.read_csv(self.train_file_path, index_col="PassengerId")
        self.test_df = pd.read_csv(self.test_file_path, index_col="PassengerId")

    def _create_social_status(self):
        for df in [self.train_df, self.test_df]:
            df["title"] = df["Name"].str.split(",|\\.", expand=True)[1]
            df["title"] = df["title"].str.strip()
            status_map = {
                "Capt": "Military",
                "Col": "Military",
                "Don": "Noble",
                "Dona": "Noble",
                "Dr": "Dr",
                "Jonkheer": "Noble",
                "Lady": "Noble",
                "Major": "Military",
                "Master": "Common",
                "Miss": "Common",
                "Mlle": "Common",
                "Mme": "Common",
                "Mr": "Common",
                "Mrs": "Common",
                "Ms": "Common",
                "Rev": "Clergy",
                "Sir": "Noble",
                "the Countess": "Noble",
            }
            # Creating the social status field based on Name - Title
            df["SocialStatus"] = df["title"].map(status_map)

            df["Title"] = df.Name.str.extract("([A-Za-z]+)\.")
            df["Title"].replace(
                [
                    "Mlle",
                    "Mme",
                    "Ms",
                    "Dr",
                    "Major",
                    "Lady",
                    "Countess",
                    "Jonkheer",
                    "Col",
                    "Rev",
                    "Capt",
                    "Sir",
                    "Don",
                ],
                [
                    "Miss",
                    "Miss",
                    "Miss",
                    "Mr",
                    "Mr",
                    "Mrs",
                    "Mrs",
                    "Other",
                    "Other",
                    "Other",
                    "Mr",
                    "Mr",
                    "Mr",
                ],
                inplace=True,
            )

    def _fill_missing_values(self):
        # Filling missing values in Age using Title
        for df in [self.train_df, self.test_df]:
            df.loc[(df.Age.isnull()) & (df.Title == "Mr"), "Age"] = df.Age[
                df.Title == "Mr"
            ].mean()
            df.loc[(df.Age.isnull()) & (df.Title == "Mrs"), "Age"] = df.Age[
                df.Title == "Mrs"
            ].mean()
            df.loc[(df.Age.isnull()) & (df.Title == "Master"), "Age"] = df.Age[
                df.Title == "Master"
            ].mean()
            df.loc[(df.Age.isnull()) & (df.Title == "Miss"), "Age"] = df.Age[
                df.Title == "Miss"
            ].mean()
            df.loc[(df.Age.isnull()) & (df.Title == "Other"), "Age"] = df.Age[
                df.Title == "Other"
            ].mean()

            df["Cabin"] = df["Cabin"].fillna("Z")
            df["CabinLetter"] = df["Cabin"].apply(lambda x: x[0])

            df["Fare"].fillna(
                df.groupby("Pclass")["Fare"].transform("mean"), inplace=True
            )
            df["Embarked"].fillna("S", inplace=True)

    def _create_new_features(self):
        for df in [self.train_df, self.test_df]:
            df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
            df["IsAlone"] = 0
            df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    def _set_label_encoding(self):
        for df in [self.train_df, self.test_df]:
            for feature in NON_NUMERIC_FEATURES:
                df[feature] = LabelEncoder().fit_transform(df[feature])

    def _set_onehot_encoding(self):
        for feature in CATEGORICAL_FEATURES:
            one_hot = pd.get_dummies(self.train_df[feature], prefix=feature)
            self.train_df = pd.concat([self.train_df, one_hot], axis=1)
            self.train_df.drop(feature, axis=1, inplace=True)

            one_hot = pd.get_dummies(self.test_df[feature], prefix=feature)
            self.test_df = pd.concat([self.test_df, one_hot], axis=1)
            self.test_df.drop(feature, axis=1, inplace=True)

    def _set_standardization(self):
        for feature in CONTINUOUS_FEATURES:
            self.train_df[[feature]] = StandardScaler().fit_transform(
                self.train_df[[feature]].to_numpy()
            )
            self.test_df[[feature]] = StandardScaler().fit_transform(
                self.test_df[[feature]].to_numpy()
            )

    def _visualize_correlation(self):
        # Compute the correlation matrix
        corr_matrix = self.train_df.corr()

        # Plot the correlation matrix using a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()

    def _feature_selection_rfe(self):
        # Feature selection using Recursive Feature Elimination (RFE)
        X = self.train_df.drop([TARGET], axis=1)
        y = self.train_df[TARGET]

        rf_classifier = RandomForestClassifier()

        # Initialize the RFE object with the Random Forest classifier
        rfe = RFE(estimator=rf_classifier, n_features_to_select=7)

        # Perform Recursive Feature Elimination
        rfe.fit_transform(X, y)

        # Get the selected feature indices
        selected_indices = rfe.get_support(indices=True)

        # Get the names of the selected features
        features_list = X.columns.values.tolist()
        selected_features = [features_list[idx] for idx in selected_indices]

        return selected_features

    def _feature_selection_correlation(self, df):
        self._visualize_correlation()
        corr_threshold = 0.1
        corr_matrix = df.corr()

        # Select features based on correlation with the target variable (Survived) if the correlation is higher than corr_threshold
        selected_features = corr_matrix[TARGET][
            abs(corr_matrix[TARGET]) > corr_threshold
        ].index
        return list(selected_features)

    def get_train_df(self):
        return self.train_df

    def get_test_df(self):
        return self.test_df

    def feature_engineering(self):
        # Perform the feature engineering steps
        self._create_social_status()
        self._fill_missing_values()
        self._create_new_features()

        for df in [self.train_df, self.test_df]:
            df.drop(UNNECESSARY_FEATURES, axis=1, inplace=True)
            df.reset_index(drop=True, inplace=True)

        self._set_label_encoding()

    def feature_selection(self, method="RFE"):
        if method == "RFE":
            feature_list = self._feature_selection_rfe()
        else:
            feature_list = self._feature_selection_correlation(self.train_df)
        self.test_df = self.test_df[feature_list].reset_index(drop=True)
        feature_list.append(TARGET)
        self.train_df = self.train_df[feature_list].reset_index(drop=True)
            
        self._set_standardization()
        self._set_onehot_encoding()
        return feature_list
