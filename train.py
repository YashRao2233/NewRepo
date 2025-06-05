import os
# MUST BE FIRST - PREVENT WINDOWS PATH CACHING
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['MLFLOW_TRACKING_URI'] = 'file:./mlruns'  # Force relative path

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Initialize MLflow with explicit relative paths
mlflow.set_tracking_uri("file:./mlruns")  # Dot for current directory
mlflow.set_experiment("Iris_Classification")

# Load data
df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_model_name = None
best_model_score = 0.0
best_model_uri = None

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model directly (no file artifacts)
        mlflow.sklearn.log_model(model, "model")
        
        if accuracy > best_model_score:
            best_model_score = accuracy
            best_model_name = model_name
            best_model_uri = f"runs:/{run.info.run_id}/model"

# Register best model
if best_model_uri:
    mlflow.register_model(best_model_uri, "Best_Iris_Classifier_Model")

print(f"Best model: {best_model_name} with accuracy: {best_model_score:.4f}")
