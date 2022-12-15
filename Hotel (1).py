from PIL import Image 
import nltk
nltk.download('wordnet')
import pandas as pd
import warnings
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from rake_nltk import Rake
import pickle
import streamlit as st
import numpy as np
from nltk.stem import PorterStemmer,WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# loading the trained model
pickle_in = open(r"rf_model.pkl", 'rb') 
model = pickle.load(pickle_in)

pickle_in = open(r"vectorizer.pkl", 'rb') 
vectorizer = pickle.load(pickle_in)

# Title of the application
st.header("𝐏𝐫𝐞𝐝𝐢𝐜𝐭 𝐑𝐚𝐭𝐢𝐧𝐠𝐬 𝐟𝐨𝐫 𝐇𝐨𝐭𝐞𝐥 𝐑𝐞𝐯𝐢𝐞𝐰𝐬")
st.subheader("𝐄𝐧𝐭𝐞𝐫 𝐓𝐡𝐞 𝐑𝐞𝐯𝐢𝐞𝐰 𝐓𝐨 𝐀𝐧𝐚𝐥𝐲𝐳𝐞")

input_text = st.text_area("Type review here", height=100)

option = st.sidebar.selectbox('Menu bar',['Sentiment Analysis','Keywords'])
st.set_option('deprecation.showfileUploaderEncoding', False)
if option == "Sentiment Analysis":
    
    
    
    if st.button("Predict sentiment"):
       
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        pickle_in = open(r"rf_model.pkl", 'rb') 
        pickle_in = open(r"rf_model.pkl", 'rb') 
        model = pickle.load(pickle_in)
        pickle_in = open(r"vectorizer.pkl", 'rb') 
        vectorizer = pickle.load(pickle_in)
        transformed_input = vectorizer.transform([text])
        
        if model.predict(transformed_input)  == -1:
            st.write(" Negative Statement 😔")
            
        elif    model.predict(transformed_input) == 1:
            st.write("Positive Statement 😃")
            #st.balloons()
        else:
            st.write(" Neutral 😶")
        

elif option == "Keywords":
    st.header("Keywords")
    if st.button("Keywords"):
        
        r=Rake(language='english') #RAKE: Rapid Automatic Keyword Extraction
        r.extract_keywords_from_text(input_text)
        # Get the important phrases
        phrases = r.get_ranked_phrases()
        # Get the important phrases
        phrases = r.get_ranked_phrases()
        # Display the important phrases
        st.write("These are the **keywords** causing the above sentiment:")
        for i, p in enumerate(phrases):
            st.write(i+1, p)
            

#displaying the image on streamlit app


import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local(r"white_flowers_bg.jpg")

st.snow()

