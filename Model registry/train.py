import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Breast_Cancer_Classification")

depths = [2,4,6,8,10]

for depth in depths:
    with mlflow.start_run():
        model = RandomForestClassifier(
            max_depth=depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("accuracy", accuracy)

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, predictions
        )
        plt.savefig("confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("confusion_matrix.png")
        mlflow.sklearn.log_model(
            model, "RandomForestModel"
        )

        print(f"Depth = {depth}     Accuracy = {accuracy: .4f}")