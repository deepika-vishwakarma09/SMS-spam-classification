
# src/evaluate.py

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from preprocess import transform_text

def evaluate():
    df = pd.read_csv(r'C:\Users\DIPIKA VISHWAKARMA\Desktop\ML-project\SMS-spam-classification\Data\Spam_SMS.csv')
    df = df[['Class', 'Message']]
    df.columns = ['target', 'text']
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    df['transformed_text'] = df['text'].apply(transform_text)

    model = joblib.load(r'C:\Users\DIPIKA VISHWAKARMA\Desktop\ML-project\SMS-spam-classification\Models\spam_classifier.pkl')
    cv = joblib.load(r'C:\Users\DIPIKA VISHWAKARMA\Desktop\ML-project\SMS-spam-classification\Models\vectorizer.pkl')

    X = cv.transform(df['transformed_text'])
    y = df['target']
    y_pred = model.predict(X)

    print("Classification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

if __name__ == "__main__":
    evaluate()
