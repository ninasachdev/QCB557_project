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

# Add ids and comment out others when using
princeton_id = 'aa8417'
#princeton_id = 'ns5404'
#princeton_id = 'jf...'

project_dir = f'/scratch/gpfs/{princeton_id}/QCB557_project'

model_name = 'fine_tune_new_v2'
model_out_dir = f'{project_dir}/models/{model_name}'

# use gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
print(config.num_labels) #2 labels
model = AutoModelForSequenceClassification.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

# don't need to move the tokenizer to gpu b/c it's light
# use data collator to pad sequences dynamically during training
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, padding=True)
tokenizer.pad_token = "X"

# default when using AutoModelForSequenceClassification is that all pretrained parameters/layers are frozen except classification head
# as we go through rounds of fine-tuning, we can optionally unfreeze from of the pretrained layers to improve performance

# unfreeze specific layers
for name, param in model.named_parameters():
    if "classifier" in name or "encoder.layer.11" in name or "encoder.layer.10" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
'''
for name, param in model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
'''

model.to(device)

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
        label = (self.dataframe.iloc[idx]['label'])

        # tokenize the sequence
        # tokenizer automatically generates attention masks
        inputs = self.tokenizer(sequence, padding='max_length', max_length=128, return_tensors='pt')
        
        # move inputs to gpu
        #inputs = {key: value.to(device) for key, value in inputs.items()}

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': (torch.tensor([label], dtype=torch.long))
        }

train_data_to_split = pd.read_csv(f'{project_dir}/data/train.csv')
train_data_to_split['label'] = train_data_to_split['label'].astype(int)

test_data = pd.read_csv(f'{project_dir}/data/test.csv')
test_data['label'] = test_data['label'].astype(int)

test_ds = custom_data_load(dataframe = test_data, tokenizer = tokenizer, shuffle=False)

train_data, valid_data = train_test_split(train_data_to_split, test_size=0.1, random_state=42)

train_ds = custom_data_load(dataframe = train_data, tokenizer = tokenizer, shuffle=True)
valid_ds = custom_data_load(dataframe = valid_data, tokenizer = tokenizer, shuffle=False)

train = DataLoader(train_ds, pin_memory=True)
valid = DataLoader(valid_ds, pin_memory=True)

#data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id='X')

training_args = TrainingArguments(
    output_dir = model_out_dir,
    num_train_epochs= 100, 
    per_device_train_batch_size=32, # powers of 2
    weight_decay=0.015, # regularization
    learning_rate=1e-5,
    gradient_accumulation_steps=2, # how many batches of gradients are accumulated before parameter update
    #gradient_checkpointing=True, # helps reduce memory
    #dataloader_num_workers=4,
    
    log_level="error",
    evaluation_strategy="steps",  
    eval_steps=500,        
    logging_steps=500,
    logging_strategy="steps",
    save_strategy="no", # don't want to save checkpoints -- takes up too much space, can change later
    fp16=True,
    #dataloader_pin_memory=True
    )

#training_args_output = f'{project_dir}/models_output/train_args_{model_name}.json'
#training_args.save_args(training_args_output)

# compute metrics (accuracy, precision, recall, f1)
def compute_metrics(eval_pred):
    logits,labels = eval_pred
    logits = torch.FloatTensor(logits[0])

    predicted_probs = torch.softmax(logits, dim=-1) # might not need this part
    predicted_labels = torch.argmax(predicted_probs, dim=-1)
    
    accuracy = accuracy_score(labels, predicted_labels)
    precision = precision_score(labels, predicted_labels)
    recall = recall_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics
    #data_collator = data_collator
)

trainer.train()
trainer.save_model(training_args.output_dir)

log_df = pd.DataFrame(trainer.state.log_history)
log_df.to_csv(f'{project_dir}/model_output/log_{model_name}.csv', index=False)

# load fine-tuned model
config = BertConfig.from_pretrained(f'{training_args.output_dir}/config.json')
print(config.num_labels) #2 labels
model = AutoModelForSequenceClassification.from_pretrained(training_args.output_dir, trust_remote_code=True, config=config)
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
        break

# concatenate predictions and convert true_labels to numpy array
predictions = torch.cat(predictions).cpu().numpy()
true_labels = np.array(true_labels)

# save evaluation results to CSV
results_df = pd.DataFrame({'true_labels': true_labels, 'predicted_labels': predictions})
results_df.to_csv(f'{project_dir}/model_output/results_{model_name}.csv', index=False)

