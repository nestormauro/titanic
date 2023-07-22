class Config:
    TARGET = "Survived"
    UNNECESSARY_FEATURES = [
        "SibSp",
        "Parch",
        "Name",
        "Title",
        "title",
        "Ticket",
        "Cabin",
    ]
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
    AVAILABLE_ALGORITHMS = ["Dummy", "RandomForest", "GradientBoosting"]
    PACKAGE_RUN_OPTIONS = ["Train", "Predict", "Test"]

settings = Config()
