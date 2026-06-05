import pickle

from fastapi import FastAPI
from pydantic import BaseModel

# Create FastAPI application
app = FastAPI()


# Load vectorizer
with open("models/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Load trained model
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)


# Request Body
class IncidentRequest(BaseModel):
    incident: str


# API Endpoint
@app.post("/predict")
def predict(request: IncidentRequest):

    incident_vector = vectorizer.transform(
        [request.incident]
    )

    prediction = model.predict(
        incident_vector
    )

    return {
        "category": prediction[0]
    }