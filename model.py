import numpy as np

CLASSES = ['biological','cardboard','glass','metal','paper','plastic','trash']

def load_model():
    from tensorflow import keras
    return keras.models.load_model("model.keras")

def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_with_probs(model, img):
    x = preprocess(img)
    preds = model.predict(x, verbose=0)[0]
    probs = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
    label = max(probs, key=probs.get)
    return label, probs

def get_suggestion(category):
    ideas = {
        "plastic": "Reuse or recycle",
        "paper": "Recycle",
        "glass": "Recycle carefully",
        "metal": "Recycle",
        "cardboard": "Reuse or recycle",
        "trash": "Dispose properly",
        "biological": "Compost"
    }
    return ideas.get(category, "No suggestion available")