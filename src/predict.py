
# src/predict.py

import joblib
from preprocess import transform_text

# Load saved model and vectorizer
model = joblib.load('models/spam_classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict_message(msg):
    # Preprocess
    transformed_msg = transform_text(msg)
    # Vectorize
    vectorized_msg = vectorizer.transform([transformed_msg])
    # Predict
    prediction = model.predict(vectorized_msg)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example usage
if __name__ == '__main__':
    user_input = input("Enter a message to check if it's spam: ")
    result = predict_message(user_input)
    print("Prediction:", result)
