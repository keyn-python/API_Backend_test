# train_model.py 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# Ensure output directory exists
os.makedirs("trained_data", exist_ok=True)

# Load your dataset
df = pd.read_csv("csv/Students Performance Dataset.csv")

# Add a Result column if not in CSV
df["Result"] = df.apply(
    lambda row: "Pass"
    if row["Study_Hours_per_Week"] >= 5 and row["Participation_Score"] >= 7 and row["Attendance (%)"] >= 80
    else "Fail",
    axis=1
)

# Select input features and label
X = df[[
    "Study_Hours_per_Week",
    "Sleep_Hours_per_Night",
    "Participation_Score",
    "Attendance (%)"
]]
y = df["Result"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "trained_data/grade_encoder.pkl")

# Save feature names
joblib.dump(X.columns.tolist(), "trained_data/model_features.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create classifier pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        max_iter=1000,
        early_stopping=True,
        random_state=42
    ))
])

# Train and save
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "trained_data/model_cls.pkl")

print("✅ Training complete. Model and encoder saved to 'trained_data/'")
