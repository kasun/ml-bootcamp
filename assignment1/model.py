import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def run():
    """Load data and run model on it."""

    df = pd.read_csv("data/processed.cleveland.data", names=["Age", "Sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"])

    # remove rows with '?' values
    mask = (df == '?').any(axis=1)
    cleaned_df = df[~mask]

    X = cleaned_df.loc[:, cleaned_df.columns != 'num']
    y = cleaned_df["num"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    predict_data = {
        "Age": 65,
        "Sex": 1.0,
        "CP": 4.0,
        "TrestBPS": 160.0,
        "Chol": 270.0,
        "fbs": 1.0,
        "RestECG": 2.0,
        "Thalach": 180.0,
        "Exang": 1.0,
        "Oldpeak": 1.5,
        "slope": 2.0,
        "ca": 3.0,
        "thal": 3.0,
    }
    print(f"Predicting using following data: {predict_data}")

    data = []
    for data_name in ["Age", "Sex", "CP", "TrestBPS", "Chol", "fbs", "RestECG", "Thalach", "Exang", "Oldpeak", "slope", "ca", "thal"]:
        data.append(predict_data[data_name])

    res = model.predict([data])
    print(f"Heart disease prediction is: {res[0]}")


if __name__ == '__main__':
    run()
