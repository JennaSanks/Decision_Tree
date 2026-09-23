import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("bank-marketing.csv")

# Remove unnecessary columns if present
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Target column
target = "y"

# Encode categorical columns
label_encoders = {}

for column in df.columns:
    if df[column].dtype == "object":
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        label_encoders[column] = encoder

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Decision Tree
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# Save model + encoders + feature names
model_data = {
    "model": model,
    "encoders": label_encoders,
    "features": list(X.columns)
}

with open("bank_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("bank_model.pkl created successfully!")