import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

CLASSES = ['biological','cardboard','glass','metal','paper','plastic','trash']

def load_model():
    return tf.keras.models.load_model("model.h5")

def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict_with_probs(model, img):
    x = preprocess(img)
    preds = model.predict(x, verbose=0)[0]
    probs = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
    label = max(probs, key=probs.get)
    return label, probs

def get_suggestion(category):
    ideas = {
        "plastic": "Reuse as plant pot or recycle.",
        "paper": "Reuse for notes or recycle.",
        "glass": "Recycle carefully to avoid injury.",
        "metal": "Sell as scrap or recycle.",
        "cardboard": "Reuse for storage or recycle.",
        "trash": "Dispose in general waste.",
        "biological": "Use for composting."
    }
    return ideas.get(category, "No suggestion available")