import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer
model = joblib.load("news-article-categories.pkl")
vectorizer = joblib.load("tfidf_vectorizer (1).pkl")

# Text preprocessing function
def preprocess(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

st.title('News Article Categorizer')
st.write('Enter a news article to categorize it.')

# Input text box
user_input = st.text_area('Enter article text here')

if st.button('Categorize'):
    if user_input.strip():
        # Preprocess the input
        user_input_processed = preprocess(user_input)
        
        # Vectorize the input
        user_input_vectorized = vectorizer.transform([user_input_processed])
        
        # Predict the category
        prediction = model.predict(user_input_vectorized)
        
        st.write(f'The predicted category is: {prediction[0]}')
    else:
        st.write('Please enter some text to categorize.')
