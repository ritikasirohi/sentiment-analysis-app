import pandas as pd
import re
import nltk
import numpy as np
import streamlit as st

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load data
amazon_df = pd.read_csv("amazon_cells_labelled.txt", delimiter='\t', header=None, names=['sentence', 'label'])
imdb_df = pd.read_csv("imdb_labelled.txt", delimiter='\t', header=None, names=['sentence', 'label'])
yelp_df = pd.read_csv("yelp_labelled.txt", delimiter='\t', header=None, names=['sentence', 'label'])

# Combine data
df = pd.concat([amazon_df, imdb_df, yelp_df], ignore_index=True)

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in filtered]
    return " ".join(lemmatized)

df['cleaned_sentence'] = df['sentence'].apply(preprocess_text)

# TF-IDF and Model
tfidf = TfidfVectorizer(ngram_range=(1, 2))
X = tfidf.fit_transform(df['cleaned_sentence'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# Prediction function
def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment

# Streamlit UI
st.title("Sentiment Analysis App")
user_input = st.text_input("Enter a sentence:")

if st.button("Predict"):
    result = predict_sentiment(user_input)
    st.write(f"**Predicted Sentiment:** {result}")
