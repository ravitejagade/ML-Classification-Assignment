
import os
import json
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# Create output folders
os.makedirs("models", exist_ok=True)

# Load dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# Save dataset
df = X.copy()
df["target"] = y
df.to_csv("dataset.csv", index=False)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test data
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("test_data.csv", index=False)

# Define five ML models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000)
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)
    ),
    "Gaussian Naive Bayes": GaussianNB(),
    "SVM": make_pipeline(
        StandardScaler(), SVC(probability=True, random_state=42)
    )
}

results = []

# Train and evaluate
for name, model in models.items():
    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "MCC": matthews_corrcoef(y_test, predictions)
    })

    filename = name.replace(" ", "_")
    joblib.dump(model, f"models/{filename}.pkl")

# Save evaluation results
results_df = pd.DataFrame(results)
results_df.to_csv("metrics.csv", index=False)

# Save feature information
with open("feature_columns.json", "w") as file:
    json.dump(list(X.columns), file)

print("\n================================")
print("ALL FIVE MODELS TRAINED SUCCESSFULLY")
print("================================")
print(results_df.round(4).to_string(index=False))
print("\nDataset records:", len(X))
print("Number of features:", X.shape[1])
print("Training and evaluation completed!")