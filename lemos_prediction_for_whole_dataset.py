# -*- coding: utf-8 -*-
"""
CS834_BERT training
lemos_prediction_for_whole_dataset.py
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from pretty_html_table import build_table
from datetime import datetime
import pytz
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras_core as keras
import keras_nlp
# import seaborn as sns
# import matplotlib.pyplot as plt
# import datetime
import sklearn
import re

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


tweets_dataset = "dataset/new_dataset_from_vstepanenko_lemos.csv"
file_root_csv = 'results/csv/'
model_path = 'saved_models/lemos_DT_nlp_bert_047.keras'  #'lemos_DT_nlp_022.keras'

prob_DIS_TRUE_threshold = 0.75
prob_DIS_FALSE_threshold = 0.75

# KERAS model loading
reloaded_model_22 = keras.models.load_model(model_path)

# DATASET loading
df = pd.read_csv(tweets_dataset)

def pre_process(text):
    text = re.sub(r"\n","",text)
    text = text.lower()
    text = re.sub(r"\d","",text)        #Remove digits
    text = re.sub(r'[^\x00-\x7f]',r' ',text) # remove non-ascii
    text = re.sub(r'[^\w\s]','',text) #Remove punctuation
    text = re.sub(r'http\S+|www.\S+', '', text) #Remove http
    return text


def process_whole_dataset():

    global df

    df_new = df[:]

    # pre-processing
    # df_new['text'] = df_new['text'].apply(lambda x : pre_process(x))    
    
    ## pre-processing NLTK
    df_new['text']=df_new['text'].apply(remove_blank)
    df_new['text']=df_new['text'].apply(contract_text)
    df_new['text']=df_new['text'].apply(handling_accented_chr)
    df_new['text']=df_new['text'].apply(clean_text)
    df_new['text']=df_new['text'].apply(lemmatization)
       

    x_test_series = df_new['text']

    predictions_df = pd.DataFrame()

    predictions_df['id'] = df_new['id']
    predictions_df['target'] = 0
    predictions_df['prob_DIS_TRUE'] = 0 # probability for the tweet being an actual disaster tweet
    predictions_df['prob_DIS_FALSE'] = 0 # probability for the tweet NOT being an actual disaster tweet
    predictions_df["prediction"] = "NOT a Disaster"

    # predictions and assigning values to columns
    predictions = reloaded_model_22.predict(x_test_series)

    predictions_df["target"] = np.argmax(predictions, axis=1)
    predictions_df['prob_DIS_TRUE'] = tf.sigmoid(predictions[:,1])
    predictions_df['prob_DIS_FALSE'] = tf.sigmoid(predictions[:,0])
    predictions_df.loc[predictions_df["target"] == 1, "prediction"] = "Disaster"

    #newly added- threshold
    # predictions_df.loc[predictions_df["prob_DIS_TRUE"] < 0.75, "target"] = 0

    result = pd.concat([df_new, predictions_df], axis=1)

    # print(result)

    result_all = result[:]
    # print('.apply(lambda x: re.sub(')
    # result_all['keyword'] = result_all['keyword'].apply(lambda x: re.sub('%20', '-', str(x)))

    result_all_ = result_all.loc[(result_all['prob_DIS_TRUE']>=prob_DIS_TRUE_threshold) | (result_all['prob_DIS_FALSE']>=prob_DIS_FALSE_threshold)]

    threshold_prefix = str(prob_DIS_TRUE_threshold)+"_"+str(prob_DIS_FALSE_threshold)
    
    
    to_csv_disasters_df = result_all_[:]

    print("\n\nnew_training_rows : ",len(to_csv_disasters_df),end="\n\n\n")

    to_csv_disasters_df = to_csv_disasters_df.iloc[:, [0,1,2,3,4,6,7,8,9]].reset_index(drop=True)

    tz_VA = pytz.timezone('America/Virgin')
    datetime_VA = datetime.now(tz_VA)

    file_name_csv = file_root_csv + threshold_prefix + "_" + datetime_VA.strftime("%y_%m_%d_%H_%M_%S")+'.csv'

    to_csv_disasters_df.to_csv(file_name_csv, index=False)


def main():
      
    process_whole_dataset()
        
    print("Successful")


if __name__ == "__main__":
    main()