"""
CS834_Fall_2023_Disaster_Tweets_Project__
|
lemos_test_app.py
Created on Wed Nov  8 17:45:29 2023
@author: Rochana Obadage
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from pretty_html_table import build_table
from datetime import datetime
from subprocess import call

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
import pyfiglet

import lemos_validate as lval



# tweets_dataset = "dataset/new_tweets_from_vstepanenko_lemos.csv"
tweets_dataset = "dataset/id_changed_new_tweets_from_vstepanenko_lemos.csv"
keyword = 'fire'
tweet_count = 100

initial_train_records_count = len(pd.read_csv(r'dataset/train_split.csv')) # 6851

file_root_html = 'results/'
file_root_csv = 'results/csv/'
model_path = 'saved_models/lemos_DT_nlp_bert_CURR.keras'

prob_DIS_TRUE_threshold = 0.7
prob_DIS_FALSE_threshold = 0.7

# KERAS model loading
reloaded_model_22 = keras.models.load_model(model_path)

# DATASET loading
df = pd.read_csv(tweets_dataset)


def merge_new_records_and_extend_training_dataset():
    folder_path_for_csv = 'results/csv/' 
    csv_file_list = glob.glob(f"{folder_path_for_csv}/*.csv")

    df_all = pd.DataFrame(columns = ['id', 'keyword', 'location', 'text', 'target', 'prob_DIS_TRUE',
           'prob_DIS_FALSE', 'prediction'])
    
    for csv_file in csv_file_list:
        df = pd.read_csv(csv_file)
        df_all = pd.concat([df_all,df]) 
    
    df_all = df_all.sort_values(by=['id'])   
    df_unique = df_all.drop_duplicates(subset=['id'])

    df_all.to_csv(r'dataset/all_extracted_data_in_one.csv',index=False)
    df_unique.to_csv(r'dataset/all_unique_data_in_one.csv',index=False)    

    # merging new records with initial training dataset
    initial_training_df = pd.read_csv('dataset/train_split.csv')
    
    # remove additional columns from unique_new_records df
    df_unique = df_unique.iloc[:,:5]   
    df_all_training = pd.concat([initial_training_df, df_unique])
    
    # initial_training_df.head()
    print('initial_training_df : ',len(initial_training_df))
    print('unique_new_records : ',len(df_unique))
    print('df_all_training : ',len(df_all_training))
    
    df_all_training.to_csv(r'dataset/extended_training_dataset.csv',index=False)


def entry_menu():
    total_characters = 76
    fonts = ['delta_corps_priest_1','cyberlarge','bubble','ansi_regular','ansi_shadow','dos_rebel','starwars','calvin_s']
    project_name = "Disaster Tweets"
    application_name = pyfiglet.figlet_format(project_name,font=fonts[3])

    print("{:^{total_characters}}".format('',total_characters=total_characters))
    print(application_name)
    print("{:_^{total_characters}}".format('',total_characters=total_characters))
    print("{:^{total_characters}}".format('',total_characters=total_characters))

    print("{:*^{total_characters}}".format('  Actual Disaster Tweets Finder  ',total_characters=total_characters))
    print("\n")    

def pre_process(text):
    text = re.sub(r"\n","",text)
    text = text.lower()
    text = re.sub(r"\d","",text)        #Remove digits
    text = re.sub(r'[^\x00-\x7f]',r' ',text) # remove non-ascii
    text = re.sub(r'[^\w\s]','',text) #Remove punctuation
    text = re.sub(r'http\S+|www.\S+', '', text) #Remove http
    return text

def visualize_disaster_tweets_at_CLI(df_cli, count):
    print("\nPRINTING THE HIGHEST RELEVANT 10 ACTUAL DISASTER TWEETS out of collected {}\n\n".format(count))

    for key,row in df_cli.iterrows():
        print(key+1,". ","id : ",row['id'],' --- ', "confidence score : ",row['prob_DIS_TRUE'])
        print(row['text'])

        print('\n')
    
        if key == 9:
            break

def process_keyword(keyword):

    # validate the newest model from previous input trigger training and check whether the F1 is better than the current model
    # call -- lemos_validate.py -- return the model path

    new_model_path = lval.model_path_for_best_model()

    if new_model_path != '':
        print("PLEASE WAIT : Loading the newly trained Best Performing Model")
        reloaded_model_22 = keras.models.load_model(new_model_path)  

    global df

    df_new = df[:]

    # pre-processing
    df_new['text_processed'] = df_new['text'].apply(lambda x : pre_process(x)) 

    # 1st 100
    # print(len(df_new))
    df_with_keyword = df_new[df_new['text'].str.contains(keyword, na=False, case=False)]

    # if len(df_with_keyword)>100:
    #     df_with_keyword = df_with_keyword[:tweet_count]

    # x_test_series = df_with_keyword['text']
    x_test_series = df_with_keyword['text_processed']

    predictions_df = pd.DataFrame()

    predictions_df['id'] = df_with_keyword['id']
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

    # removing processed column
    df_with_keyword.drop(columns='text_processed',inplace=True)

    result = pd.concat([df_with_keyword, predictions_df], axis=1)

    # print(result)

    result_all = result[:]
    # print('.apply(lambda x: re.sub(')
    result_all['keyword'] = result_all['keyword'].apply(lambda x: re.sub('%20', '-', str(x)))

    result_all_ = result_all.loc[(result_all['prob_DIS_TRUE']>=prob_DIS_TRUE_threshold) | (result_all['prob_DIS_FALSE']>=prob_DIS_FALSE_threshold)]


    
    to_csv_disasters_df = result_all_[:]

    to_csv_disasters_df = to_csv_disasters_df.iloc[:, [0,1,2,3,5,6,7,8]].reset_index(drop=True)
    
    to_html_disasters_df = result_all.iloc[:, [0,1,2,3,6,7,8]].reset_index(drop=True)
    # print(to_html_disasters_df.head())
    # to_html_disasters_df.sort_values(by = ['prob_DIS_TRUE', 'prob_DIS_FALSE'], ascending = [False, False],inplace=True)[:tweet_count]
    to_html_disasters_df = to_html_disasters_df.sort_values(by = ['prob_DIS_TRUE', 'prob_DIS_FALSE'], ascending = [False, False])
    
    for_CLI_actual_disasters_df_for_user = to_csv_disasters_df[to_csv_disasters_df['target']==1]
    for_CLI_actual_disasters_df_for_user = for_CLI_actual_disasters_df_for_user.sort_values(by=['prob_DIS_TRUE'], ascending=False).reset_index(drop =True)

    tz_VA = pytz.timezone('America/Virgin')
    datetime_VA = datetime.now(tz_VA)

    html_table_blue_light = build_table(to_html_disasters_df[:tweet_count], 'blue_light',font_family='Trebuchet MS',padding="1px 2px 1px 4px") #'Helvetica', Georgia

    file_name = file_root_html+keyword+"_" + datetime_VA.strftime("%y_%m_%d_%H_%M_%S")+'.html'
    file_name_csv = file_root_csv+keyword+"_" + datetime_VA.strftime("%y_%m_%d_%H_%M_%S")+'.csv'

    to_csv_disasters_df.to_csv(file_name_csv, index=False)

    visualize_disaster_tweets_at_CLI(for_CLI_actual_disasters_df_for_user, len(for_CLI_actual_disasters_df_for_user))

    # for_CLI_actual_disasters_df_for_user #this is for CLI
    with open(file_name, 'w') as f:
        f.write(html_table_blue_light)


    # submit the new training job with model_name and iteration??
    current_train_records_count = len(pd.read_csv(r'dataset/extended_training_dataset.csv')) # 6851

    if int(1.1*(initial_train_records_count)) < current_train_records_count:
        
        itr = lval.get_next_iteration()
        epochs = '4'
        dynamic_vals = f"--export=ALL,iteration={itr},epochs={epochs}"
        out_filename = f"out_files_from_slurm/lemos_834_{itr}.txt"
        
        call(["sbatch", dynamic_vals, "-o", out_filename, "training_job_submission.sh"])


def main():
    x = 'y'
    while(True):
    
        # print("Hello Lemos\n")
        entry_menu()
        tweet_key = input("*****  Enter the disaster keyword : ")
        print("Processing...\n|\n")
        
        process_keyword(tweet_key)

        print("{:_^{total_characters}}".format('',total_characters=76))
        x = input('Do you want to continue (y or n) :: ')
        
        if x == 'n':
            break
    print("\n")


if __name__ == "__main__":
    main()