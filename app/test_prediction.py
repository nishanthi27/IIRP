import pickle

with open("models/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

incident = "unauthorized login attempt"

incident_vector = vectorizer.transform([incident])

prediction = model.predict(incident_vector)

print(prediction[0])