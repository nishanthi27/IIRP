import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/incidents.csv")

X = df["text"]

y_category = df["category"]

y_severity = df["severity"]

X_train, X_test, y_cat_train, y_cat_test = train_test_split(
    X,
    y_category,
    test_size=0.2,
    random_state=42,
    stratify=y_category
)

_, _, y_sev_train, y_sev_test = train_test_split(
    X,
    y_severity,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)

X_test_vectorized = vectorizer.transform(X_test)

# Category Model
category_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

category_model.fit(
    X_train_vectorized,
    y_cat_train
)

# Severity Model
severity_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

severity_model.fit(
    X_train_vectorized,
    y_sev_train
)

cat_predictions = category_model.predict(
    X_test_vectorized
)

sev_predictions = severity_model.predict(
    X_test_vectorized
)

print(
    "Category Accuracy:",
    accuracy_score(
        y_cat_test,
        cat_predictions
    )
)

print(
    "Severity Accuracy:",
    accuracy_score(
        y_sev_test,
        sev_predictions
    )
)

with open("models/vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

with open("models/category_model.pkl", "wb") as file:
    pickle.dump(category_model, file)

with open("models/severity_model.pkl", "wb") as file:
    pickle.dump(severity_model, file)

print("Training Completed")