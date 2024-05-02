import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification, 
    DataCollatorForTokenClassification, 
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer, 
    DataCollatorForLanguageModeling
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from transformers.models.bert.configuration_bert import BertConfig 
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
#from training_args_module import training_args
import transformers
import os

import csv
from Bio import SeqIO


import argparse

princeton_id = 'aa8417'
project_dir = f'/scratch/gpfs/{princeton_id}/QCB557_project'

# use gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_out_dir = '/scratch/gpfs/aa8417/QCB557_project/models/replicate_043024/rep1/fine_tune_parallel_v1'

config_seq = BertConfig.from_pretrained(f'/scratch/gpfs/{princeton_id}/QCB557_project/models/replicate_043024/rep1/fine_tune_parallel_v1/config.json', output_attentions=True)
print(config_seq.num_labels) #2 labels
model_seq = AutoModelForSequenceClassification.from_pretrained(model_out_dir, trust_remote_code=True, config=config_seq)
model_seq.to(device)

config_base = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M", output_hidden_states=True, output_attentions=True)
print(config_base.num_labels) #2 labels
model_base = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config_base)
model_base.to(device)

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, padding=True)
tokenizer.pad_token = "X"

test = pd.read_csv(f'/scratch/gpfs/{princeton_id}/QCB557_project/data/H3K4me3/test.csv')
train = pd.read_csv(f'/scratch/gpfs/{princeton_id}/QCB557_project/data/H3K4me3/train.csv')
full = pd.concat([train, test], ignore_index=True)
full.to_csv(f'/scratch/gpfs/{princeton_id}/QCB557_project/data/H3K4me3/all_seqs.csv')

def get_model_output(model_base, model_seq, tokenizer, dataframe, device):
    results_dict = {}

    for index, row in dataframe.iterrows():
        sequence = row['sequence']
        label = row['label']
        
        inputs = tokenizer(sequence, padding='max_length', max_length=128, return_tensors='pt').to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs_seq = model_seq(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            hidden_states = outputs_seq.hidden_states
        
        with torch.no_grad():
            outputs_base = model_base(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            attention_weights = outputs_base.attentions

        key = f'seq_{index}_{label}'
        results_dict[key] = {
            "sequence": sequence,
            "input_ids": input_ids.tolist(),
            "hidden_states": hidden_states,
            "attention_weights": attention_weights
        }

    return results_dict

results_dict = get_model_output(model_base, model_seq, tokenizer, full, device)

json_file_path = f'/scratch/gpfs/{princeton_id}/QCB557_project/data/H3K4me3/results_dict.json'

with open(json_file_path, 'w') as json_file:
    json.dump(results_dict, json_file)