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



# def read_f1_score(df,model_path):
#     # def displayConfusionMatrix(y_true, y_pred, dataset):
#     # disp = ConfusionMatrixDisplay.from_predictions(
#     #     y_true,
#     #     np.argmax(y_pred, axis=1),
#     #     display_labels=["Not Disaster","Disaster"],
#     #     cmap=plt.cm.Blues
#     # )

#     y_true = df['target???']
#     y_pred = df['label']

#     tn, fp, fn, tp = confusion_matrix(y_true, np.argmax(y_pred, axis=1)).ravel()
#     f1_score = tp / (tp+((fn+fp)/2))

#     disp.ax_.set_title("Confusion Matrix on " + dataset + " Dataset -- F1 Score: " + str(f1_score.round(2)))    
#     f1 = 0.7

#     return f1


def validate_new_model():
    new_model_path = get_recent_model_path()
    
    curr_f1 = 0.8
    new_f1 = 0.7

    # write logs
    with open('saved_models/model_performances.txt','a') as f:
        f.write(new_model_path)
        f.write(new_f1)

    return new_f1


def model_path_for_best_model(model_name):
    
    # 'saved_models/lemos_DT_nlp_bert_044.keras'
    model_path = ''
    
    if curr_f1 < new_f1: # implement this part
        model_path = get_recent_model_path()
        
        # copying the best model to be lemos_DT_nlp_bert_CURR.keras
        curr_file_full_path = "saved_models/lemos_DT_nlp_bert_CURR.keras"
        # best_model_path = "saved_models/test.txt"
        
        dest = shutil.copy(model_path, curr_file_full_path)

    return model_path