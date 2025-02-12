import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # lowercase
    text = nltk.word_tokenize(text)  # tokenisation

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)  # removing special characters
    # text = y it is wrong as string is mutable. We can use cloning
    text = y[:]  # cloning
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
trf_fn = pickle.load(open('trf_fn.pkl','rb'))

st.title("Email/SMS Spam classifier")

input_sms = st.text_input("Enter the message")

# Add Button

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # transformed_sms = trf_fn(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
