
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import joblib
from src.preprocess import transform_text

# Load model and vectorizer
model = joblib.load('models/spam_classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

st.title(" SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # Preprocess
    transformed = transform_text(input_sms)
    vectorized = vectorizer.transform([transformed]).toarray()
    
    # Predict
    result = model.predict(vectorized)[0]

    # Output
    if result == 1:
        st.error(" It's a SPAM message!")
    else:
        st.success(" It's a HAM (Not Spam) message.")
