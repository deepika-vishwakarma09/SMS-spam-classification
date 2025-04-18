
# src/train.py

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from preprocess import transform_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['Class', 'Message']]
    df.columns = ['target', 'text']
    df.drop_duplicates(inplace=True)

    # Encode labels
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})

    # Preprocess
    df['transformed_text'] = df['text'].apply(transform_text)
    return df

def train_model(df):
    import os
    os.makedirs("models", exist_ok=True)

    cv = CountVectorizer()
    X = cv.fit_transform(df['transformed_text']).toarray()
    y = df['target']

    # 🔁 Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # ✅ Evaluate
    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)

    # 🔍 Confusion matrix & classification report
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    #  Save confusion matrix plot
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    #  Save model and vectorizer
    joblib.dump(model, 'models/spam_classifier.pkl')
    joblib.dump(cv, 'models/vectorizer.pkl')

    #  Log with MLflow
    mlflow.start_run()
    mlflow.log_param("model", "MultinomialNB")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("test_accuracy", acc_score)
    mlflow.log_artifact("models/spam_classifier.pkl")
    mlflow.log_artifact("models/vectorizer.pkl")
    mlflow.log_artifact("models/confusion_matrix.png")
    mlflow.end_run()

    print(f"Model trained and evaluated. Test Accuracy: {acc_score:.4f}")


if __name__ == '__main__':
    filepath = r'C:\Users\DIPIKA VISHWAKARMA\Desktop\ML-project\SMS-spam-classification\Data\Spam_SMS.csv'
    df = load_and_prepare_data(filepath)
    train_model(df)
