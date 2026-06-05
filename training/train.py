import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================
# STEP 1: LOAD DATASET
# ==========================

df = pd.read_csv("data/incidents.csv")

print("Dataset Loaded Successfully")
print(df.head())


# ==========================
# STEP 2: FEATURES & LABELS
# ==========================

X = df["text"]
y = df["category"]


# ==========================
# STEP 3: TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining Records: {len(X_train)}")
print(f"Testing Records: {len(X_test)}")


# ==========================
# STEP 4: TF-IDF
# ==========================

vectorizer = TfidfVectorizer()

# Learn vocabulary from training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Use same vocabulary on test data
X_test_vectorized = vectorizer.transform(X_test)


# ==========================
# STEP 5: TRAIN MODEL
# ==========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_vectorized, y_train)

print("\nModel Training Completed")


# ==========================
# STEP 6: EVALUATE MODEL
# ==========================

predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# ==========================
# STEP 7: SAVE VECTORIZER
# ==========================

with open("models/vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nVectorizer Saved")


# ==========================
# STEP 8: SAVE MODEL
# ==========================

with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Saved")


print("\nTraining completed successfully")