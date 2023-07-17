import json
import pandas as pd
from copy import deepcopy

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from transformers import BertForQuestionAnswering

# Tokenizer
model_checkpoint = "mrm8488/bert-multi-cased-finetuned-xquadv1"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# ========== Read Data ==========


class SQUADDataset:
    def __init__(self, set_name, file_path) -> None:
        assert set_name in ['train', 'dev', 'test']
        self.set_name = set_name
        self.file_path = file_path
    
    def get_data(self):
        raw = json.load(open(self.file_path, 'r'))
        data = []
        for article in raw['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    _ = {
                        'context':context,
                        'question':qa['question'],
                        'answers':qa['answers']
                    }
                    data.append(_)
        self.df = pd.DataFrame(data)
    
    def add_end_index(self):
        new_answers = []
        for idx, row in self.df.iterrows():
            context = row['context']
            _ = []
            for answer in row['answers']:
                gold_text = answer['text']
                start_idx = answer['answer_start']
                end_idx = start_idx + len(gold_text)

                if context[start_idx:end_idx] == gold_text:
                    answer['answer_end'] = end_idx
                    _.append(answer)
                else:
                    for n in [1, 2]:
                        if context[start_idx-n:end_idx-n] == gold_text:
                            answer['answer_start'] = start_idx - n
                            answer['answer_end'] = end_idx - n
                            _.append(answer)
            new_answers.append(_)
        
        self.df['answers'] = new_answers
        self.df = self.df\
            .explode('answers')\
            .rename(columns={'answers':'answer'})
        
    def tokenization(self, tokenizer):
        
        train_contexts = self.df['context'].tolist()
        train_questions = self.df['question'].tolist()
        train_answers = self.df['answer'].tolist()
        
        encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)

        # initialize lists to contain the token indices of answer start/end
        start_positions = []
        end_positions = []
        for i in range(len(train_answers)):
            # append start/end token position using char_to_token method
            start_positions.append(encodings.char_to_token(i, train_answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, train_answers[i]['answer_end']))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            # end position cannot be found, char_to_token found space, so shift position until found
            shift = 1
            while end_positions[-1] is None:
                end_positions[-1] = encodings.char_to_token(i, train_answers[i]['answer_end'] - shift)
                shift += 1
        # update our encodings object with the new token-based start/end positions
        if len(start_positions) == 0:
            start_positions.append(512)
        if len(end_positions) == 0:
            end_positions.append(512)
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        
        return encodings

# ========== 定義 Dataset ==========
import torch

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # print(self.encodings)
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


# ========== Training ==========
def train(model, train_dataset, dev_dataset):

    # setup GPU/CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model over to detected device
    model.to(device)
    # activate training mode of model
    model.train()
    # initialize adam optimizer with weight decay (reduces chance of overfitting)
    optim = AdamW(model.parameters(), lr=1e-5)
    best_val_loss = 999999

    # initialize data loader for training data
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dev_dataset, batch_size=24, shuffle=False)

    for epoch in range(10):
        # set model to train mode
        model.train()
        # setup loop (we use tqdm for the progress bar)
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all the tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            # train model on batch and return outputs (incl. loss)
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            # extract loss
            loss = outputs[0]
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
        
        # setup loop (we use tqdm for the progress bar)
        model.eval()
        val_loop = tqdm(val_loader, leave=True)
        val_loss = 0
        for batch in val_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            # extract loss
            loss = outputs[0]
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            val_loop.set_description(f'Epoch {epoch} val loss')
            val_loop.set_postfix(loss=loss.item())
            val_loss += loss.item()
        
        # save model
        if val_loss/len(val_loader) < best_val_loss:
            best_val_loss = val_loss/len(val_loader)
            torch.save(model.state_dict(), f'output/{model_checkpoint}-drcd-qa.bin')
            model.config.to_json_file(f'output/{model_checkpoint}-drcd-qa.bin')
            tokenizer.save_vocabulary('output')
            print("save this model ---------------> \n")
    
    return None

if __name__ == '__main__':

    # Training Set
    drcd_train = SQUADDataset('train', 'datasets/DRCD/DRCD_training.json')
    drcd_train.get_data()
    drcd_train.add_end_index()
    drcd_train_encodings = drcd_train.tokenization(tokenizer)

    # Dev Set
    drcd_dev = SQUADDataset('dev', 'datasets/DRCD/DRCD_dev.json')
    drcd_dev.get_data()
    drcd_dev.add_end_index()
    drcd_dev_encodings = drcd_dev.tokenization(tokenizer)

    # build datasets for both our training and validation sets
    train_dataset = SquadDataset(drcd_train_encodings)
    dev_dataset = SquadDataset(drcd_dev_encodings)
    
    model = BertForQuestionAnswering.from_pretrained(model_checkpoint)
    train(model, train_dataset, dev_dataset)