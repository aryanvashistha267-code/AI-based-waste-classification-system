import streamlit as st
from PIL import Image
import pandas as pd
from model import load_model, predict_with_probs, get_suggestion

st.set_page_config(page_title="AI Waste Classifier", page_icon="♻️")

st.title("♻️ AI Waste Classifier")
st.caption("Smart waste classification system")

model = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", use_container_width=True)

    label, probs = predict_with_probs(model, img)

    confidence = max(probs.values())
    top2 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]

    if confidence < 0.5:
        label_display = "Uncertain"
    elif (top2[0][1] - top2[1][1]) < 0.15:
        label_display = f"{top2[0][0]} / {top2[1][0]}"
    else:
        label_display = label

    st.subheader(f"Detected: {label_display}")
    st.write(f"Confidence: {round(confidence*100,2)}%")

    st.progress(float(confidence))

    df = pd.DataFrame({
        "Category": list(probs.keys()),
        "Probability": list(probs.values())
    })

    st.bar_chart(df.set_index("Category"))

    st.subheader("Suggestion")
    st.info(get_suggestion(label))