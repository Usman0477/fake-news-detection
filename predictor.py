"""Prediction helper used by the Streamlit interface."""
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fake_news_model.joblib"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. First run: python train_model.py"
        )
    return joblib.load(MODEL_PATH)

def predict_news(model, news_text):
    text = (news_text or "").strip()
    if len(text) < 40:
        raise ValueError("Please enter a headline or article with at least 40 characters.")

    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    fake_probability = float(probabilities[classes.index(0)])
    real_probability = float(probabilities[classes.index(1)])

    label = "FAKE" if fake_probability >= real_probability else "REAL"
    confidence = max(fake_probability, real_probability)

    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": fake_probability,
        "real_probability": real_probability
    }
