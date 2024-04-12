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
from transformers.models.bert.configuration_bert import BertConfig 
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import transformers
#import Dataset
import os

import csv
from Bio import SeqIO

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt

#os.environ['WANDB_NOTEBOOK_NAME'] = 'fine-tune'

# Add ids and comment out others when using
princeton_id = 'aa8417'
#princeton_id = 'ns5404'
#princeton_id = 'jf...'

project_dir = f'/scratch/gpfs/{princeton_id}/QCB557_project'

model_name = 'fine_tune_v2'
model_out_dir = f'{project_dir}/models/{model_name}'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# load another fine-tuned model
model_name = 'fine_tune_new_v0'
model_load = f'{project_dir}/models/{model_name}'
config = BertConfig.from_pretrained(f'{model_load}/config.json')
model = AutoModelForSequenceClassification.from_pretrained(model_load, trust_remote_code=True, config=config)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, padding=True)
tokenizer.pad_token = "X"

class custom_data_load(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, shuffle=True):
        if shuffle:
            self.dataframe = dataframe.sample(frac=1).reset_index(drop=True)  # shuffle the dataframe
        else:
            self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sequence = self.dataframe.iloc[idx]['sequence']
        #print(sequence)
        #print(len(sequence))
        label = (self.dataframe.iloc[idx]['label'])
        #print(label)

        # tokenize the sequence
        # tokenizer automatically generates attention masks
        #inputs = self.tokenizer(sequence, padding='max_length', max_length=500, truncation=True, return_tensors='pt')
        inputs = self.tokenizer(sequence, padding='max_length', max_length=128, return_tensors='pt')
        #print(inputs)
        
        # move inputs to gpu
        #inputs = {key: value.to(device) for key, value in inputs.items()}

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': (torch.tensor([label], dtype=torch.long))
            #'labels': torch.tensor([int(label)]) 
        }

test_data = pd.read_csv(f'{project_dir}/data/test.csv')
test_data['label'] = test_data['label'].astype(int)

test_ds = custom_data_load(dataframe = test_data, tokenizer = tokenizer, shuffle=False)

model.eval()

predictions = []
true_labels = []

# go through test dataset, make predictions and store them
for idx, sample in enumerate(test_ds):
    with torch.no_grad():
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        predicted_probs = torch.softmax(outputs.logits, dim=-1) # might not need this part
        predicted_labels = torch.argmax(outputs.logits, dim=-1)

        predictions.append(predicted_labels)
        true_labels.append(sample['labels'].item())
        break

# concatenate predictions and convert true_labels to numpy array
predictions = torch.cat(predictions).cpu().numpy()
true_labels = np.array(true_labels)

# save evaluation results to CSV
results_df = pd.DataFrame({'true_labels': true_labels, 'predicted_labels': predictions})
results_df.to_csv(f'{project_dir}/model_output/results_{model_name}.csv', index=False)