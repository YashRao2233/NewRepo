import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Set experiment name
mlflow.set_experiment("Iris_Classification")

# Load data (use relative path)
data_path = os.path.join("data", "iris.csv")
df = pd.read_csv(data_path)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Ensure models directory exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

best_model_name = None
best_model_score = 0.0
best_model_uri = None

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)

        # Save model locally in relative path
        model_filename = f"{model_name}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)

        # Log artifact with relative path
        mlflow.log_artifact(model_path, artifact_path="models")

        # Log model to MLflow model registry format
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Track best model
        if acc > best_model_score:
            best_model_score = acc
            best_model_name = model_name
            best_model_uri = f"runs:/{run.info.run_id}/model"

# Register best model (optional)
if best_model_uri:
    result = mlflow.register_model(
        model_uri=best_model_uri,
        name="Best_Iris_Model"
    )
    print(f"Registered Best Model: {best_model_name} with accuracy: {best_model_score:.4f}")
