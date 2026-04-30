import streamlit as st
from PIL import Image
import pandas as pd
from model import load_model, predict_with_probs, get_suggestion

st.set_page_config(page_title="AI Waste Classifier", page_icon="♻️", layout="centered")

st.title("♻️ AI Waste Classifier")
st.caption("Smart multi-class waste detection with AI")

model = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", width="stretch")

    label, probs = predict_with_probs(model, img)

    confidence = max(probs.values())
    top2 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]

    if confidence < 0.5:
        label_display = "Uncertain"
        status = "low"
    elif (top2[0][1] - top2[1][1]) < 0.15:
        label_display = f"{top2[0][0]} / {top2[1][0]}"
        status = "medium"
    else:
        label_display = label
        status = "high"

    st.subheader(f"🧠 Detected: {label_display}")

    st.write("Confidence")
    st.progress(float(confidence))
    st.write(f"{round(confidence*100,2)}%")

    if status == "high":
        st.success("High confidence")
    elif status == "medium":
        st.warning("Close prediction between classes")
    else:
        st.error("Low confidence – try clearer image")

    st.subheader("📊 Prediction Distribution")

    df = pd.DataFrame({
        "Category": list(probs.keys()),
        "Probability": list(probs.values())
    })

    st.bar_chart(df.set_index("Category"))

    st.subheader("🔍 Top Predictions")
    for k, v in top2:
        st.write(f"{k} → {round(v*100,2)}%")

    st.subheader("💡 Suggestion")
    st.info(get_suggestion(label))