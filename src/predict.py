import pickle
from src.preprocess import clean_text

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]