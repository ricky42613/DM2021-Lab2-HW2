from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from torch.utils.data import DataLoader
import torch
import csv
from tqdm import tqdm

torch.cuda.empty_cache()
MODEL_NAME = 'bert-base-uncased'

class Predictor():
    def __init__(self,model_path="./dl_final.bin"):
        self.label_list = ['sadness','disgust','anticipation','joy','trust','anger','fear','surprise']
        self.model = torch.load(model_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def tokenize_function(self,data):
        return self.tokenizer(data['text'], padding="max_length", truncation=True)
    
    def predict(self,text):
        tokenize_data = self.tokenizer(text, padding="max_length", truncation=True)
        for k in tokenize_data:
            tokenize_data[k] = torch.as_tensor([tokenize_data[k]]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tokenize_data['input_ids'],tokenize_data['attention_mask'])
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            return predictions.tolist()[0]
    
    def prdict_csv(self,test_file,output_file):
        datas = []
        cnt = 0
        test_datasets = load_dataset('csv',delimiter='\t', data_files=test_file)
        output_f = csv.writer(open(output_file, 'w'))
        output_f.writerow(['id','emotion'])
        for d in test_datasets['train']:
            datas.append(d['ID'])
        test_datasets_tokenize = test_datasets.map(self.tokenize_function,batched=True)
        test_datasets_tokenize = test_datasets_tokenize.remove_columns(["text"])
        test_datasets_tokenize = test_datasets_tokenize.remove_columns(["ID"])
        test_datasets_tokenize.set_format("torch")
        test_data = test_datasets_tokenize['train']
        test_dataloader = DataLoader(test_data, batch_size=8)
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                for pred in predictions:
                    output_f.writerow([datas[cnt],self.label_list[pred.item()]])
                    cnt += 1



if __name__ == "__main__":
    predictor = Predictor(model_path="./dl_final.bin")
    predictor.prdict_csv(test_file="miss.csv",output_file="missBert.csv")

    
