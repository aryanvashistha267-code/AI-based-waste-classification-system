# AI Waste Classifier ♻️

A simple AI system that classifies waste (plastic, paper, glass, metal, cardboard)
and provides reuse/recycling suggestions.

## How it works
- Uses MobileNetV2 (pretrained on ImageNet) for feature extraction
- Adds a small classification head for 5 waste categories
- Provides rule-based suggestions after classification

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Notes
- This demo uses a pretrained backbone without fine-tuning
- Accuracy depends on image quality and similarity to training data