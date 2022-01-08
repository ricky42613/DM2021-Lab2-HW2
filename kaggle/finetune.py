from datasets import load_dataset,load_metric
from transformers import AutoTokenizer,TrainingArguments,Trainer,AutoModelForSequenceClassification,AdamW,get_scheduler
import numpy as np
from torch.utils.data import DataLoader
import torch
import logging
MODEL_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
logger = logging.getLogger(__name__)

# torch.cuda.empty_cache()


def tokenize_function(data):
    # print(tokenizer(data["text"], padding="max_length", truncation=True))
    return tokenizer(data["text"], padding="max_length", truncation=True)

print(tokenize_function({'text':'hello world'}))

train_datasets,eval_datasets = load_dataset('csv', data_files='train.csv',split=['train[:50000]', 'train[50000:60000]'])
print(train_datasets)
print(eval_datasets)

train_datasets_tokenize = train_datasets.map(tokenize_function,batched=True)
eval_datasets_tokenize = eval_datasets.map(tokenize_function,batched=True)



train_datasets_tokenize = train_datasets_tokenize.remove_columns(["text"])
train_datasets_tokenize = train_datasets_tokenize.remove_columns(["ID"])
train_datasets_tokenize.set_format("torch")

eval_datasets_tokenize = eval_datasets_tokenize.remove_columns(["text"])
eval_datasets_tokenize = eval_datasets_tokenize.remove_columns(["ID"])
eval_datasets_tokenize.set_format("torch")

# train_dataset = train_datasets_tokenize["train"]
# eval_dataset = train_datasets_tokenize["test"]
# print(type(train_dataset))
train_dataloader = DataLoader(train_datasets_tokenize, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(eval_datasets_tokenize, batch_size=8)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=8)
optimizer = AdamW(model.parameters(), lr=1e-5)

num_epochs = 8
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
# progress_bar = tqdm(range(num_training_steps))
metric= load_metric("accuracy")

for epoch in range(num_epochs):
    print(epoch)
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        # progress_bar.update(1)
    
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    eval_metric = metric.compute()
    print(f"epoch {epoch}: {eval_metric}")
    logger.info(f"epoch {epoch}: {eval_metric}")

torch.save(model, "./dl_final.bin")