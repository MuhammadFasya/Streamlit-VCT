import streamlit as st
import joblib
import re
from textblob import TextBlob

# Load the TF-IDF Vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load the Sentiment Model (Random Forest)
model = joblib.load("sentiment_model_rf.pkl")


label_map_inverse = {0: "positive", 1: "neutral", 2: "negative"}

st.title("VCT Reddit Comment Sentiment Analyzer")

user_input = st.text_area("Enter a VCT Reddit comment:", "")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

if st.button("Analyze Sentiment"):
    if user_input:
        preprocessed_input = preprocess_text(user_input)

        # Transform the preprocessed text using the loaded TF-IDF vectorizer
        input_vector = vectorizer.transform([preprocessed_input])

        # Make prediction
        prediction_encoded = model.predict(input_vector)[0]
        predicted_sentiment = label_map_inverse.get(prediction_encoded, "unknown")

        st.write(f"**Predicted Sentiment:** {predicted_sentiment}")
    else:
        st.write("Please enter a comment to analyze.")
