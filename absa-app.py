import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# List of aspects
aspects = [
    "service", "staff", "crew", "attitude", "luggage", "baggage", "lost", "claim",
    "flight", "delay", "cancellation", "late", "boarding", "process", "gate", "check-in",
    "seat", "comfort", "legroom", "space", "food", "meal", "snack", "drink",
    "price", "cost", "value", "money", "cleanliness", "dirty", "hygiene",
    "entertainment", "wifi", "tv", "screen", "communication", "update", "notice",
    "ground service", "check in", "in-flight entertainment", "cabin crew", "flight path",
    "flight delay", "refund policy", "rebooking", "online check-in"
]

# Function to extract aspects
def extract_aspects(text, aspect_list):
    text = text.lower()
    matched = set()
    for aspect in aspect_list:
        pattern = r'\b' + re.escape(aspect.lower().replace('-', ' ')) + r's?\b'
        if re.search(pattern, text):
            matched.add(aspect)
    return list(matched)

# Load general sentiment model
@st.cache_resource
def load_general_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load ABSA model & tokenizer
@st.cache_resource
def load_absa_model():
    tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    return tokenizer, model

# Aspect-level sentiment
def analyze_aspect_sentiment(review, aspect, tokenizer, model):
    inputs = tokenizer(f"[CLS] {review} [SEP] {aspect} [SEP]", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0].numpy()
    sentiment = ["negative", "neutral", "positive"][probs.argmax()]
    return sentiment, float(probs.max())

# Streamlit UI
st.title("🧠 Aspect-Based Sentiment Analyzer")

text = st.text_area("✏️ Enter a customer review:", height=200)

if st.button("🔍 Analyze"):
    if not text.strip():
        st.error("Please enter a review.")
    else:
        with st.spinner("Loading models and analyzing..."):
            general_model = load_general_model()
            tokenizer, absa_model = load_absa_model()

            # General Sentiment
            general_result = general_model(text)[0]
            st.subheader("Overall Sentiment")
            st.write(f"**Sentiment**: {general_result['label']} (Confidence: {general_result['score']:.4f})")

            # Aspect Detection
            aspects_found = extract_aspects(text, aspects)
            st.subheader("🔍 Detected Aspects")
            if aspects_found:
                for asp in aspects_found:
                    asp_sentiment, conf = analyze_aspect_sentiment(text, asp, tokenizer, absa_model)
                    st.write(f"**{asp}** → {asp_sentiment.capitalize()} (Confidence: {conf:.4f})")
            else:
                st.info("No known aspects found in the text.")

# Sidebar
st.sidebar.markdown("## ℹ️ How to use")
st.sidebar.write("1. Enter a review mentioning things like service, food, delay, etc.")
st.sidebar.write("2. The app detects aspects and returns specific sentiments.")