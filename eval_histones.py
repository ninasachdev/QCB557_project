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

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
histones = ['H3', 'H3K14ac', 'H3K36me3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H4', 'H4ac']
histone = histones[idx]

# put best model path here
model_path = ''

# load fine-tuned model
config = BertConfig.from_pretrained(f'{training_args.output_dir}/config.json')
print(config.num_labels) #2 labels
model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True, config=config)
model.to(device)

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

# concatenate predictions and convert true_labels to numpy array
predictions = torch.cat(predictions).cpu().numpy()
true_labels = np.array(true_labels)

# save evaluation results to CSV
results_df = pd.DataFrame({'true_labels': true_labels, 'predicted_labels': predictions})
results_df.to_csv(f'{project_dir}/model_output/results_{model_name}_{histone}.csv', index=False)