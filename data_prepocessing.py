import json
from pathlib import Path
import torch

"!pip install transformers # solo una volta da terminale!"

"""

"""
def read_squad(path):
  path = Path(path)
  with open(path, 'rb') as f:
      squad_dict = json.load(f)

  contexts = []
  questions = []
  answers = []
  for group in squad_dict['data']:
      for passage in group['paragraphs']:
          context = passage['context']
          for qa in passage['qas']:
              question = qa['question']
              for answer in qa['answers']:
                  contexts.append(context)
                  questions.append(question)
                  answers.append(answer)

  return contexts, questions, answers


"""

"""
def add_end_idx(answers, contexts):
  for answer, context in zip(answers, contexts):
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)

    # sometimes squad answers are off by a character or two – fix this
    if context[start_idx:end_idx] == gold_text:
        answer['answer_end'] = end_idx
    elif context[start_idx-1:end_idx-1] == gold_text:
        answer['answer_start'] = start_idx - 1
        answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
        answer['answer_start'] = start_idx - 2
        answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters



"""

"""
def add_token_positions(encodings, answers, tokenizer):
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
    # if None, the answer passage has been truncated
    if start_positions[-1] is None:
        start_positions[-1] = tokenizer.model_max_length
    if end_positions[-1] is None:
        end_positions[-1] = tokenizer.model_max_length
  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


"""
SquadDataset Class 
"""
class SquadDataset(torch.utils.data.Dataset): 
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
