import streamlit as st
import joblib
import re

# Load model dan vectorizer
model = joblib.load("sentiment_model_rf.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Mapping label prediksi
label_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Fungsi untuk bersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# UI Streamlit
st.set_page_config(page_title="VCT Reddit Sentiment Analyzer")
st.title("🔍 VCT Reddit Comment Sentiment Analyzer")
st.write("Masukkan komentar seperti di Reddit, dan sistem akan memprediksi sentimennya (positif, netral, atau negatif).")

user_input = st.text_area("📝 Komentar Reddit", "")

if st.button("Prediksi Sentimen"):
    if user_input:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        sentiment = label_mapping.get(pred, "Unknown")

        st.success(f"Hasil Prediksi: **{sentiment}** 🎯")
    else:
        st.warning("Masukkan komentar terlebih dahulu.")
