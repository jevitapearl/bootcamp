import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not installed. Skipping Target Encoding.")


def main():

    print("Loading dataset")

    file_path = "electricity.csv"

    if not os.path.exists(file_path):
        print(f"Error: cannot find {file_path}")
        return

    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully. Rows: {df.shape[0]}, Features: {df.shape[1]}")

    # Missing values
    print("Cleaning categorical columns")

    df["day"] = df["day"].astype(str).str.replace("b'", "").str.replace("'", "")
    df["class"] = df["class"].astype(str).str.replace("b'", "").str.replace("'", "")

    print("Checking missing values")

    imputer = SimpleImputer(strategy="median")

    numeric_columns = ["date", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    print("Missing values after imputation:", df.isnull().sum().sum())


    # Feature Engineering
    print("Creating new features")

    df["price_difference"] = (df["nswprice"] - df["vicprice"])
    df["demand_difference"] = (df["nswdemand"] - df["vicdemand"])

    #Target Encoding

    encoder = LabelEncoder()
    df["class_encoded"] = encoder.fit_transform(df["class"])
    print("Classes:",encoder.classes_)

    if TargetEncoder:
        print("Applying Target Encoding")
        target_encoder = TargetEncoder()
        df["day_encoded"] = target_encoder.fit_transform(df["day"],df["class_encoded"])

    else:
        print("Using Label Encoding for day")
        df["day_encoded"] = encoder.fit_transform(df["day"])


    # Feature Selection

    features = ["date","period","nswprice","nswdemand","vicprice","vicdemand","transfer","price_difference","demand_difference","day_encoded"]
    X_features = df[features]
    y = df["class_encoded"]
    selector = SelectKBest(score_func=mutual_info_classif,k=5)
    selector.fit(X_features,y)
    selected = selector.get_support()
    best_features = (X_features.columns[selected].tolist())

    print("Best Features:", best_features)

    # Train Test Split
    X = df[best_features]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    print("Training size:",X_train.shape)

    print("Testing size:",X_test.shape)

    # Model Training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)

    # Predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test,predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Show examples
    for i in range(5):
        actual = encoder.inverse_transform([y_test.iloc[i]])[0]
        predicted = encoder.inverse_transform([predictions[i]])[0]

        print("\n")
        print("Model guessed:", predicted)
        print("Real answer:", actual)



if __name__ == "__main__":
    main()