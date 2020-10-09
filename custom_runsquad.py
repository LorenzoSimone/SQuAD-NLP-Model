import data_prepocessing
from data_prepocessing import SquadDataset
from data_prepocessing import add_token_positions, add_end_idx, read_squad
from transformers import AutoTokenizer, ElectraForQuestionAnswering, ElectraConfig, AdamW, ElectraTokenizerFast
from torch.utils.data import DataLoader
from transformers import AdamW
from run_squad import evaluate
from types import SimpleNamespace
import argparse
import torch
import itertools
import os

tokenizer = ElectraTokenizerFast.from_pretrained('deepset/electra-base-squad2')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EPOCHS = 3
BATCH_SIZE = 16
RESULTS_FOLDER = 'results'

eval_args = {
   "model_type":"electra",
   "model_name_or_path":"deepset/electra-base-squad2",
   "output_dir":"output",
   "data_dir":"squad",
   "train_file": None,
   "predict_file":"valid_squadv2.json",
   "config_name":"",
   "tokenizer_name":"",
   "cache_dir":"",
   "version_2_with_negative": True,
   "null_score_diff_threshold":0.0,
   "max_seq_length":384,
   "doc_stride":128,
   "max_query_length":64,
   "do_train":False,
   "do_eval":True,
   "evaluate_during_training":False,
   "do_lower_case":True,
   "per_gpu_train_batch_size":8,
   "per_gpu_eval_batch_size":16,
   "learning_rate":5e-05,
   "gradient_accumulation_steps":1,
   "weight_decay":0.0,
   "adam_epsilon":1e-08,
   "max_grad_norm":1.0,
   "num_train_epochs":3.0,
   "max_steps":-1,
   "warmup_steps":0,
   "n_best_size":20,
   "max_answer_length":30,
   "verbose_logging":False,
   "lang_id":0,
   "logging_steps":500,
   "save_steps":500,
   "eval_all_checkpoints":False,
   "no_cuda":False,
   "overwrite_output_dir":False,
   "overwrite_cache":False,
   "seed":42,
   "local_rank":-1,
   "fp16":False,
   "fp16_opt_level":"O1",
   "server_ip":"",
   "server_port":"",
   "threads":1,
   "n_gpu":0,
   "device": device
}

eval_args = SimpleNamespace(**eval_args)

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
hyperparameters = [eval(args.learning_rate), eval(args.size_hidden), eval(args.num_hidden_layer)]
configurations = [i for i in itertools.product(*hyperparameters)]

j = 1

res_path = '/{}'.format(RESULTS_FOLDER)
try:
  os.mkdir(res_path) 
except FileExistsError:
  print("Directory " , res_path ,  " already exists")

for conf in configurations:

  conf_id = 'config{}_{}_{}_{}'.format(j, *conf)
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
  for epoch in range(EPOCHS):
    for batch in train_loader:
        if(bi > 2):
          break
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss_records.append(loss)
        loss.backward()
        optim.step()
        bi+=1


  model.eval()
  print(evaluate(eval_args, model, tokenizer))
  model.save_pretrained(conf_id)
  tokenizer.save_pretrained(conf_id)

  j += 1
