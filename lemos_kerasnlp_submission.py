import os
os.environ['KERAS_BACKEND'] = 'tensorflow'


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras_core as keras
import keras_nlp
import re

import argparse
import sys

from nltk.tokenize import word_tokenize,sent_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,LancasterStemmer

from contractions import fix 
from unidecode import unidecode

# remove newlines , spaces
def remove_blank(data):
    text=data.replace("\\n"," ").replace("\t"," ")
    return text

# Contractions mapping
def contract_text(data):
    text=fix(data)
    return text

# handling accented character
def handling_accented_chr(data):
    text=unidecode(data)
    return text

# remove stopwords
stopwords_list=stopwords.words('english')
stopwords_list.remove('no')
stopwords_list.remove('nor')
stopwords_list.remove('not')

# clean the text
def clean_text(data):
    # tokenization
    tokens=word_tokenize(data)
    # lowercase the data
    #lowercase=[word.lower() for word in tokens]
    # remove stopwords
    #remove_stopwords=[word for word in lowercase if word not in stopwords_list]
    # remove punctuations
    #remove_punctuations=[word for word in remove_stopwords if word not in punctuation]
    # remove len(word)<2
    #remove_len_word=[word for word in remove_punctuations if len(word)>2]
    # word contains only alphabet not number
    #final_text=[word for word in remove_len_word if word.isalpha()]
    
    clean_data=[i.lower() for i in tokens if (i.lower() not in punctuation) and (i.lower() not in stopwords_list) and (len(i)>2) and (i.isalpha())]

    
    return clean_data

def lemmatization(data):
    final_text=[]
    lemmatizing=WordNetLemmatizer()
    for i in data:
        lemma=lemmatizing.lemmatize(i)
        final_text.append(lemma)
        
    return " ".join(final_text)

#-----------------------------------------------

def pre_process(text):
    text = re.sub(r"\n","",text)
    text = text.lower()
    text = re.sub(r"\d","",text)        #Remove digits
    text = re.sub(r'[^\x00-\x7f]',r' ',text) # remove non-ascii
    text = re.sub(r'[^\w\s]','',text) #Remove punctuation
    text = re.sub(r'http\S+|www.\S+', '', text) #Remove http
    return text
    

def create_submission(model,submission_):


    # load_model_path = "saved_models/lemos_DT_nlp_bert_011.keras"
    load_model_path = model
    print("\n\nusing model: ",load_model_path)
    # submission_file_name = 'submission/submission_11.csv'
    submission_file_name = submission_
    read_sample_submission = 'submission/sample_submission.csv'
    
    print("TensorFlow version:", tf.__version__)
    print("keras_core version:", keras.__version__)
    print("KerasNLP version:", keras_nlp.__version__)
    print("NumPy version:", np.__version__)
    # print("sklearn version:", sklearn.__version__)
    print("pandas version:", pd.__version__)
    # print("h5py version:", h5py.__version__)
    
    
    
    df_train = pd.read_csv("dataset/train.csv")
    df_test = pd.read_csv("dataset/test.csv")
    
    print('Training Set Shape = {}'.format(df_train.shape))
    print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
    print('Test Set Shape = {}'.format(df_test.shape))
    print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))
    
    
    df_train["length"] = df_train["text"].apply(lambda x : len(x))
    df_test["length"] = df_test["text"].apply(lambda x : len(x))
    
    print("Train Length Stat")
    print(df_train["length"].describe())
    print()
    
    print("Test Length Stat")
    print(df_test["length"].describe())
    EPOCHS = 4 #6
    AUTO = tf.data.experimental.AUTOTUNE
    
    
    
    classifier = keras.models.load_model(load_model_path)

    #pre-processing method 1
    # df_test['text'] = df_test['text'].apply(lambda x : pre_process(x))

    # pre-processing NLTK
    df_test['text']=df_test['text'].apply(remove_blank)
    df_test['text']=df_test['text'].apply(contract_text)
    df_test['text']=df_test['text'].apply(handling_accented_chr)
    df_test['text']=df_test['text'].apply(clean_text)
    df_test['text']=df_test['text'].apply(lemmatization)
    
    X_test = df_test["text"]
    
    
    sample_submission = pd.read_csv(read_sample_submission)
    sample_submission.head()
    sample_submission["target"] = np.argmax(classifier.predict(X_test), axis=1)
    
    
    sample_submission.to_csv(submission_file_name, index=False)
    
    print("Successfully saved at :\n",submission_file_name)




def main():

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-model", "--model_path", help="model_path")
    argParser.add_argument("-submission", "--submission_path", help="submission_path")
    args = argParser.parse_args()

    mdl = args.model_path
    subm = args.submission_path
    
    create_submission(mdl,subm)
    

if __name__ == "__main__":
    main()






