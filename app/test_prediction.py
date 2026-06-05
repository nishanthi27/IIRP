import pickle

with open("models/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("models/category_model.pkl", "rb") as file:
    model_category = pickle.load(file)


with open("models/severity_model.pkl", "rb") as file:
    model_severity = pickle.load(file)

incident = "unauthorized login attempt"

incident_vector = vectorizer.transform([incident])

prediction_category = model_category.predict(incident_vector)
prediction_severity = model_severity.predict(incident_vector)

print(prediction_category,prediction_severity)