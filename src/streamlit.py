import streamlit as st 
from pickle import dump, load
import nltk #pip install nltk , library to clean text
from nltk import download
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import regex as re
import string


with (open("/workspaces/NLP_Spam_detection/models/vectoriser.pk", "rb")) as openfile:
    vector = load(openfile)
with (open("/workspaces/NLP_Spam_detection/models/svc_c80_ovo_degr14_seed42.pk", "rb")) as openfile:
    model = load(openfile)

def url_transform(link):
    link=link.lower() #converting to lowercase
  #segmenting url by punctuation marks: !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    link=re.sub(f'[{re.escape(string.punctuation)}]', ' ', link) #removes punctuation and adds a space
    link=re.sub(r'\s+', " ", link) #removes white spaces
    link=link.split() #splits link content
    return link

download("wordnet")
lemmatizer=WordNetLemmatizer()

download('stopwords')
stopwords=stopwords.words('english')

def nltk_funct(link, lemmatizer=lemmatizer):
#simplify words with lemmatizer
    link=[lemmatizer.lemmatize(token) for token in link]
#removing stopwords
    link=[token for token in link if token not in stopwords]

    return link

st.title('Welcome to the URL spam detector!')

url=st.text_input('Please enter below the full URL you want to check')


if st.button("Check"):
    row=nltk_funct(url_transform(url))
    row_tr=[' '.join(word) for word in row]
    vector_row=vector.transform(row_tr)
    y_pred=model.predict(vector_row)[0]
    if y_pred==False:
        st.text(str('The URL is not spam'))
    elif y_pred==True:
        st.text(str('The URL is spam'))