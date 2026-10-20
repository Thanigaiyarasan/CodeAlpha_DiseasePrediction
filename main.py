# CodeAlpha - Disease Prediction Project

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load dataset
print("Loading breast cancer dataset...")

data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

print("Total records:", X.shape[0])
print("Total features:", X.shape[1])
print("Classes:", data.target_names)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 3. Create models
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])
}

results = []

# 4. Train and evaluate
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=data.target_names,
        zero_division=0
    ))

    # Save confusion matrix
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=data.target_names,
        cmap="Blues"
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(f"{filename}_confusion_matrix.png")
    plt.close()

# 5. Save results
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison.csv", index=False)

results_df.to_csv("results_summary.csv", index=False)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

# 6. Save classification information
with open("results_summary.txt", "w") as file:
    file.write("Disease Prediction - Model Results\n\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\nEducational use only. Not for medical diagnosis.")

# 7. Plot model accuracy
plt.figure(figsize=(8, 5))

plt.bar(results_df["Model"], results_df["Accuracy"])

plt.ylabel("Accuracy")
plt.title("Disease Prediction - Model Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("model_accuracy.png")
plt.close()

print("\nProject completed successfully!")
print("Check the generated output files.")

print(
    "\nDisclaimer: This model is for educational purposes only "
    "and must not be used for real medical diagnosis."
)