import data_prepocessing
from data_prepocessing import SquadDataset
from data_prepocessing import add_token_positions, add_end_idx, read_squad
from transformers import AutoTokenizer, ElectraForQuestionAnswering, ElectraConfig, AdamW, ElectraTokenizerFast
from torch.utils.data import DataLoader
from transformers import AdamW
from types import SimpleNamespace
import argparse
import torch
import itertools
import os
import json
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

tokenizer = ElectraTokenizerFast.from_pretrained('deepset/electra-base-squad2')
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
EPOCHS = 3
BATCH_SIZE = 16
RESULTS_FOLDER = 'results'
LOSS_FILE = 'losses.json'
IMAGE_FOLDER = 'img'

# Building the parser syntax 
parser = argparse.ArgumentParser(description='Use me if you want load and preprocessing the custom squad data')

#Configuration parameters
parser.add_argument('-lr', '--learning_rate', type=str, metavar='', required=True, help='insert the learning rate possible values')
parser.add_argument('-hs', '--size_hidden', type=str, metavar='', required=True, help='insert the hidden size possible values')
parser.add_argument('-hl', '--num_hidden_layer', type=str, metavar='', required=True, help='insert the number of hidden layers possible values')

#File path parameters
parser.add_argument('-path_tr_f', '--path_trainingData_file', type=str, metavar='', required=True, help='insert the path of training custom squad dataset')
parser.add_argument('-path_vl_f', '--path_validationData_file', type=str, metavar='', required=True, help='insert the path of validation custom squad dataset')
args = parser.parse_args()

args.learning_rate = [eval(item) for item in args.learning_rate.split(',')]
args.size_hidden = [eval(item) for item in args.size_hidden.split(',')]
args.num_hidden_layer = [eval(item) for item in args.num_hidden_layer.split(',')]

#Data preparation code
train_contexts, train_questions, train_answers = read_squad(args.path_trainingData_file)
val_contexts, val_questions, val_answers = read_squad(args.path_validationData_file)

add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

tokenizer = ElectraTokenizerFast.from_pretrained('deepset/electra-base-squad2')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

add_token_positions(train_encodings, train_answers, tokenizer)
add_token_positions(val_encodings, val_answers, tokenizer)

train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)


#Constructing list of hyperparameters configurations

hyperparameters = [args.learning_rate, args.size_hidden, args.num_hidden_layer]
configurations = [i for i in itertools.product(*hyperparameters)]

j = 1

res_path = './{}/{}'.format(RESULTS_FOLDER,LOSS_FILE)
img_path = './{}/{}'.format(RESULTS_FOLDER,IMAGE_FOLDER)
os.makedirs(os.path.dirname(res_path), exist_ok=True)
os.mkdir(img_path)

with open(res_path, "w") as f:
    json.dump({'title': 'Grid search | Available Hyperparameters [learn_rate, hidden size, hidden layer] | {}'.format(configurations), 'results': []}, f)

for conf in configurations:

  conf_id = '{}_config_{}_{}_{}'.format(j, *conf)
  print('Run [{}] --learning rate: {}, --hidden size: {}, --hidden layer: {}'.format(j, *conf))

  # Initializing a ELECTRA electra-base-uncased style configuration
  electra_conf = ElectraConfig(hidden_size = conf[1], num_hidden_layers = conf[2])
  # Initializing a model from the electra-base-uncased style configuration
  model = ElectraForQuestionAnswering(electra_conf)
  configuration = model.config

  model.to(device)
  model.train()

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  learning_rate = conf[0]
  optim = AdamW(model.parameters(), lr=learning_rate)
  loss_records = []
  bi = 0

  for epoch in tqdm(range(EPOCHS)):
    for batch in train_loader:
        if(bi > 3):
          break
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss_records.append(loss.item())
        loss.backward()
        optim.step()
        bi+=1

  model.save_pretrained(conf_id)
  tokenizer.save_pretrained(conf_id)

  with open(res_path, "r") as f:
    records = json.load(f)
  with open(res_path, "w") as f:
    new_rec = {'conf_id': conf_id, 'loss': loss_records}
    json.dump({'title': records['title'], 'results': [*records['results'], new_rec]}, f)

  plt.figure()
  plt.plot(loss_records)
  plt.title('Learning curve {}'.format(conf_id))
  plt.savefig('./{}/{}.png'.format(img_path,conf_id))
  plt.close()

  j += 1
