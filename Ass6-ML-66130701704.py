
import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

model_bay = pickle.load(open('naive_bayes-66130701704.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer-66130701704.sav', 'rb'))
st.title("Review Sentiment Prediction using Naive Bayes")
user_input = st.text_input("Enter your review:")
user_input_vec = vectorizer.transform([user_input])
pred = model_bay.predict(user_input_vec)

st.write("## Prediction Result:")
st.write('Sentiment:', pred[0])
