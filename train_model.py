import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


DATA_FILE = "landslide_data.csv"
MODEL_FILE = "landslide_model.joblib"

features = [
    "rainfall",
    "soil_moisture",
    "vibration",
    "slope"
]

target = "risk"


data = pd.read_csv(DATA_FILE)

X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


model.fit(
    X_train,
    y_train
)


predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"Test accuracy: {accuracy:.2%}"
)

print(
    classification_report(
        y_test,
        predictions
    )
)


joblib.dump(
    model,
    MODEL_FILE
)

print(
    f"Model saved to {MODEL_FILE}"
)
