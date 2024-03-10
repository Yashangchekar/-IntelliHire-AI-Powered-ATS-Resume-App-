# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:04:42 2024

@author: yash
"""


import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer


tf=pickle.load(open('C:/Users/yash/Documents/NLPclass/project/Ats_resume/vectorizer.pkl','rb'))
model1=pickle.load(open('C:/Users/yash/Documents/NLPclass/project/Ats_resume/model.pkl','rb'))
#text preprocessing:-
#remove stopwords
#list of stopwords
stop=stopwords.words('english') #user define object
def clean_text(text):           #here clean_text() is a user define passing argument function
    #converting lower case  and then tokenize it
    token=word_tokenize(text.lower())
    #filter only the alphabet use inbuilt functions isalpha() #remove number and special character from text
    word_token=[t for t in token if t.isalpha()] #word token user define list object
    #remove stopwords
    #use list comphrension
    print(token)
    print(word_token)
    clean_tokens=[t for t in word_token if t not in stop] 
    
    print(clean_tokens)
    #next step of preprocessing :Lemmanitzation
    
    #create object of WordnetLemmanitzater class
    lemma=WordNetLemmatizer()
    lemmatized_token=[lemma.lemmatize(t) for t in clean_tokens]
    return " ".join(lemmatized_token)


def main():
    st.title("ATS AI Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
            
    # If UTF-8 decoding fails, try decoding with 'latin-1'
    

        cleaned_resume = clean_text(resume_text)
        input_features = tf.transform([cleaned_resume])
        prediction_id = model1.predict(input_features)[0]
        st.write(prediction_id)

        # Map category ID to category name
        # Map category ID to category name
        category1 = {
            0: "Advocate",
            1: "Arts",
            2: "Automation Testing",
            3: "Blockchain",
            4: "Business Analyst",
            5: "Civil Engineer",
            6: "Data Science",
            7: "Database",
            8: "DevOps Engineer",
            9: "DotNet Developer",
            10: "ETL Developer",
            11: "Electrical Engineering",
            12: "HR",
            13: "Hadoop",
            14: "Health and fitness",
            15: "Java Developer",
            16: "Mechanical Engineer",
            17: "Network Security Engineer",
            18: "Operations Manager",
            19: "PMO",
            20: "Python Developer",
            21: "SAP Developer",
            22: "Sales",
            #23: "Testing",
            24: "Web Designing",
         }
        category_name = category1.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)



# python main
if __name__ == "__main__":
    main()