import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer

import pandas as pd

# Download dependencies (run once)
nltk.download('vader_lexicon')
nltk.download('punkt')

# --- APP CONFIG ---
st.set_page_config(page_title="Bible Sentiment Analyzer", page_icon="📖", layout="centered")
st.title("📖 Bible-Based Sentiment Analyzer")
st.markdown("Analyze the emotional tone of a Bible verse using AI.")

# --- INITIALIZE SENTIMENT ANALYZER ---
sia = SentimentIntensityAnalyzer()

# --- USER INPUT ---
verse = st.text_area("Enter a Bible verse or passage:", height=150)

if st.button("🔍 Analyze Sentiment") and verse.strip():
    # Tokenize by sentence
    from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

tokenizer = PunktSentenceTokenizer()
sentences = tokenizer.tokenize(verse)
    
data = []
for sent in sentences:
        scores = sia.polarity_scores(sent)
        data.append({"Sentence": sent, **scores})

df = pd.DataFrame(data)
st.write("### 📊 Sentiment Breakdown")
st.dataframe(df, use_container_width=True)

    # Get overall
overall = sia.polarity_scores(verse)
    
st.write("### 🧠 Overall Sentiment")
st.metric("Positivity", f"{overall['pos']:.2f}")
st.metric("Neutrality", f"{overall['neu']:.2f}")
st.metric("Negativity", f"{overall['neg']:.2f}")
st.metric("Compound Score", f"{overall['compound']:.2f}")

if overall['compound'] >= 0.05:
        st.success("🌟 Sentiment: Positive - A joyful and hopeful passage!")
elif overall['compound'] <= -0.05:
        st.error("⚠️ Sentiment: Negative - A sorrowful or serious tone.")
else:
        st.info("😐 Sentiment: Neutral - A balanced emotional tone.")