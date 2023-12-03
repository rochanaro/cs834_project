"""
CS834_Fall_2023_Disaster_Tweets_Project__
|
lemos_validate.py
Created on Wed Nov  8 17:45:29 2023
@author: Rochana Obadage
"""

# import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import re
import glob
import os
import shutil


CURRENT_MODEL = 'saved_models/lemos_DT_nlp_bert_CURR.keras'
initial_train_records_count = len(pd.read_csv(r'dataset/train_split.csv')) # 6851
VALIDATION_DATASET = 'dataset/validation_split.csv'

print(initial_train_records_count)


def get_next_iteration():
    folder_path_for_keras = 'saved_models' 
    keras_models_list = glob.glob(f"{folder_path_for_keras}/*.keras")
    keras_models_list.sort(reverse=True)

    path = keras_models_list[1]
    val = int(path[path.index("bert_")+5:path.index(".keras")])+1
    iteration = '{:0>3}'.format(val)

    return iteration
    

def get_recent_model_path():
    folder_path_for_keras = 'saved_models' 
    keras_models_list = glob.glob(f"{folder_path_for_keras}/*.keras")
    keras_models_list.sort(reverse=True)

    return keras_models_list[1]


def get_last_2_items_from_model_perfromance():
    model_perform_file = 'results/performances/model_performances.txt'
    
    with open(model_perform_file) as f:
        content = f.read()
        
        # print((content.split("\n\n")))
    
        _2nd_last_item = content.split("\n\n")[-3]
        last_item = content.split("\n\n")[-2]
    
        # print(_2nd_last_item.split("\n"))
        # print(last_item.split("\n"))

    return _2nd_last_item.split("\n"),last_item.split("\n")


def get_f1_for_last_2_items():
    
    item_list_2nd_last,item_list_last = get_last_2_items_from_model_perfromance()

    f1_last = float(item_list_last[-1])
    f1_2nd_last = float(item_list_2nd_last[-1])
        
    return f1_2nd_last,f1_last


def get_training_records_count_for_training_job():
    
    _,item_list_last = get_last_2_items_from_model_perfromance()

    training_records_count = int(item_list_last[-2].split(':')[-1])
        
    return training_records_count


# def validate_new_model():
#     new_model_path = get_recent_model_path()
    
#     curr_f1 = 0.8
#     new_f1 = 0.7

#     # write logs
#     with open('saved_models/model_performances.txt','a') as f:
#         f.write(new_model_path)
#         f.write(new_f1)

#     return new_f1

def get_current_loaded_model_details():
    model_perform_file = 'results/performances/current_model_details.txt'
    
    with open(model_perform_file) as f:
        content = f.read()
        
        content_list = content.split("\n")

    model_path = content_list[1]
    records_used_for_training = int(content_list[2].split(":")[1])
    f1_score_local_validation = float(content_list[3])
    
        # last_item = content.split("\n\n")[-2]
    
    return model_path,records_used_for_training,f1_score_local_validation


def get_last_item_from_model_perfromance():
    model_perform_file = 'results/performances/model_performances.txt'
    
    with open(model_perform_file) as f:
        content = f.read()
        
        # print((content.split("\n\n")))
    
        last_item = content.split("\n\n")[-2].split("\n")
        print(last_item)
        model_path = last_item[2]
        records_used_for_training = int(last_item[3].split(":")[1])
        f1_score_local_validation = float(last_item[4])
    
        # print(_2nd_last_item.split("\n"))
        # print(last_item.split("\n"))

    return model_path,records_used_for_training,f1_score_local_validation


def write_last_item_details_to_current_model_details_txt():
    model_perform_file = 'results/performances/model_performances.txt'
    
    with open(model_perform_file) as f:
        content = f.read()
        
        current_model_details_file = 'results/performances/current_model_details.txt'
        last_item = content.split("\n\n")[-2].replace("\n","",1)

        with open(current_model_details_file,'w') as f1:
            f1.write(last_item)


def model_path_for_best_model():
    
    # 'saved_models/lemos_DT_nlp_bert_044.keras'
    model_path = ''

    # current loaded model
    # model_path,records_used_for_training,f1_score_local_validation
    _, _, curr_model_f1 = get_current_loaded_model_details()
    
    # newly trained latest model
    new_model_path, _, latest_f1 = get_last_item_from_model_perfromance()

    curr_f1 = 0.8
    new_f1 = 0.7
    
    if curr_model_f1 < latest_f1: # implement this part
        # changing the current model details text with to latest trained model
        write_last_item_details_to_current_model_details_txt()
        
        model_path = new_model_path
     
        # # copying the best model to be lemos_DT_nlp_bert_CURR.keras
        # curr_file_full_path = "saved_models/lemos_DT_nlp_bert_CURR.keras"
        # # best_model_path = "saved_models/test.txt"
        
        # dest = shutil.copy(model_path, curr_file_full_path)

    return model_path