#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres gpu:1

enable_lmod
module load container_env tensorflow-gpu/2.12.0

export CUDA_HOME=/cm/shared/applications/cuda-toolkit/11.7.1/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

crun.tensorflow-gpu -p ~/envs/cs834_project python lemos_kerasnlp_submission.py -model saved_models/lemos_DT_nlp_bert_031.keras -submission submission/submission_31.csv

crun.tensorflow-gpu -p ~/envs/cs834_project python lemos_kerasnlp_submission.py -model saved_models/lemos_DT_nlp_bert_032.keras -submission submission/submission_32.csv

crun.tensorflow-gpu -p ~/envs/cs834_project python lemos_kerasnlp_submission.py -model saved_models/lemos_DT_nlp_bert_033.keras -submission submission/submission_33.csv

crun.tensorflow-gpu -p ~/envs/cs834_project python lemos_kerasnlp_submission.py -model saved_models/lemos_DT_nlp_bert_034.keras -submission submission/submission_34.csv

crun.tensorflow-gpu -p ~/envs/cs834_project python lemos_kerasnlp_submission.py -model saved_models/lemos_DT_nlp_bert_035.keras -submission submission/submission_35.csv

crun.tensorflow-gpu -p ~/envs/cs834_project python lemos_kerasnlp_submission.py -model saved_models/lemos_DT_nlp_bert_036.keras -submission submission/submission_36.csv

crun.tensorflow-gpu -p ~/envs/cs834_project python lemos_kerasnlp_submission.py -model saved_models/lemos_DT_nlp_bert_037.keras -submission submission/submission_37.csv

crun.tensorflow-gpu -p ~/envs/cs834_project python lemos_kerasnlp_submission.py -model saved_models/lemos_DT_nlp_bert_038.keras -submission submission/submission_38.csv

crun.tensorflow-gpu -p ~/envs/cs834_project python lemos_kerasnlp_submission.py -model saved_models/lemos_DT_nlp_bert_039.keras -submission submission/submission_39.csv