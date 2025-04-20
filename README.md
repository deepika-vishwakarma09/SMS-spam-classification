
# SMS Spam Classification with Streamlit

This project is a SMS Spam Classification application built using Python, Machine Learning, and Streamlit. The app classifies incoming SMS messages as either spam or ham (non-spam). The app provides an easy-to-use interface for real-time classification using a trained machine learning model.


## Technologies Used:

 -Python (version: 3.x)

 -Streamlit (for app development)

 -Scikit-learn (for machine learning model)

 -Pandas (for data manipulation)

 -Numpy (for numerical operations)

 -Matplotlib/Seaborn (for visualization)

 -Pickle (for saving the trained model)

## Features

Real-time Spam Classification: 

                 Input an SMS and the model will classify it as spam or ham.

Model Performance:

find 97% accuracy by using naibe bayes.

View metrics like accuracy and confusion matrix to assess the performance of the model.



## Installation

Clone the repository:

```bash
  git clone https://github.com/deepika-vishwakarma09/SMS-spam-classification.git
  cd sms-spam-classification
```
Create a virtual environment:

```bash
  python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```
Install required dependencies:

```bash
 pip install -r requirements.txt

```
Usage:
To run the app locally, use the following command:
```bash
streamlit run app.py


```

## Model Training:

1.The machine learning model is trained on a dataset of SMS messages with labels indicating whether the message is spam or not.

2.TThe model is trained using algorithms like Logistic Regression , Naive Bayes ,SVM, ensemble learning,and xgboost(these modes check under notebook then apply nive Bayes in our project).

3.The trained model is saved using Pickle and loaded in the Streamlit app for making predictions.



## LLicense:
This project is licensed under the MIT License.