import pickle

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open("models/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("models/category_model.pkl", "rb") as file:
    category_model = pickle.load(file)

with open("models/severity_model.pkl", "rb") as file:
    severity_model = pickle.load(file)


class IncidentRequest(BaseModel):
    incident: str


@app.post("/predict")
def predict(request: IncidentRequest):

    vector = vectorizer.transform(
        [request.incident]
    )

    category = category_model.predict(
        vector
    )[0]

    severity = severity_model.predict(
        vector
    )[0]

    return {
        "category": category,
        "severity": severity
    }