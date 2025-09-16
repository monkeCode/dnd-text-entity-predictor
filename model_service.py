from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import torch
from model import NERModel
from transformers import AutoTokenizer
import yaml
import sys
import re

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

CONFIG_PATH = "./params.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)



MODEL_PATH = sys.argv[1]
BASE_MODEL_NAME =  config["base-model"]
LOWER = config["lower_texts"]

SPLIT_BY_SENTENSES = True

with open('data/ner_dataset/labels.txt', 'r') as f:
        label_list = [line.strip() for line in f]

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}
NUM_LABELS = len(label_list)


model:NERModel = NERModel(BASE_MODEL_NAME, NUM_LABELS, id2label, label2id, 0, use_prev_label=config["train"]["use_prev_label"], weights=config["train"]["weights"])
model.load_state_dict(state_dict=torch.load(MODEL_PATH))
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

class Item(BaseModel):
    batch: List[Dict[str, str]]

@torch.no_grad()
def predict(text, max_length=200, overlap=0):
    # Токенизируем весь текст для получения смещений
    full_inputs = tokenizer(text, return_offsets_mapping=True, truncation=False, return_tensors="pt")
    full_offsets = full_inputs["offset_mapping"].squeeze(0).tolist()
    
    # Инициализируем массивы для агрегации предсказаний
    all_logits = torch.zeros(len(full_offsets), NUM_LABELS)
    count = torch.zeros(len(full_offsets))
    
    # Определяем размер чанка и перекрытия
    chunk_size = max_length - 2  # -2 для [CLS] и [SEP]
    stride = int(chunk_size * (1 - overlap))
    
    # Обрабатываем текст чанками
    for start in range(0, len(full_offsets), stride):
        end = start + chunk_size
        chunk_text = text[full_offsets[start][0]:full_offsets[min(end, len(full_offsets)-2)][1]]
        
        # Токенизируем чанк
        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            max_length=max_length
        )
        
        # Получаем предсказания для чанка
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logits = outputs["logits"].squeeze(0)
        
        # Сопоставляем предсказания с исходными токенами
        chunk_offsets = inputs["offset_mapping"].squeeze(0).tolist()
        for i, (chunk_start, chunk_end) in enumerate(chunk_offsets):
            if chunk_start == chunk_end == 0:  # Специальные токены
                continue
            
            # Находим соответствующий токен в полной последовательности
            original_idx = start + i
            if original_idx >= len(all_logits):
                continue
                
            all_logits[original_idx] += logits[i]
            count[original_idx] += 1

    # Усредняем предсказания
    #averaged_logits = all_logits / count.unsqueeze(1)
    averaged_logits = all_logits
    preds = torch.argmax(averaged_logits, dim=-1).tolist()
    
    # Формируем результат
    tokens = tokenizer.convert_ids_to_tokens(full_inputs["input_ids"].squeeze(0))
    return list(zip(tokens, preds, full_offsets))

def exist_next(preds:list, label:str, index, max=3):
    for i in range(index, min(len(preds), index + max) ):
        if preds[i][1] == 0:
            continue
        if preds[i][1] % 2 == 1:
            return False
        
        return id2label[preds[i][1]][2:] == label
    
    return False

        

@app.post('/predict')
def predict_batch(item: Item):
    results = []
    for request in item.batch:
        text = request["text"]
        if LOWER:
            text = text.lower()
        if SPLIT_BY_SENTENSES:
            sentens = re.split(r'(?<=[\n\.])(?=\S)', text)
            tokens_with_preds = []
            offset = 0
            for s in sentens:
                if len(s) == 0:
                    continue
                pred = predict(s)
                tokens_with_preds.extend([(p[0], p[1], (p[2][0] + offset, p[2][1] + offset)) for p in pred])
                offset += len(s)
        else:
            tokens_with_preds = predict(text)

        id = request["id"]

        predictions = []
        current_entity = None
        for idx, (token, pred, (start_char, end_char)) in enumerate(tokens_with_preds):
            if pred != 0:
                if pred % 2 != 0:
                    if current_entity is not None:
                        current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                        predictions.append(current_entity)
                        current_entity = None
                    else:
                        current_entity = {
                        "text": "",
                        "label": model.hparams.id2label[pred][2:],
                        "start": start_char,
                        "end": end_char
                    }
                elif current_entity is not None:
                    current_entity["end"] = end_char
            else:
                if current_entity is not None:
                    if token[0] == "#" or exist_next(tokens_with_preds, current_entity["label"], idx+1):
                        current_entity["end"] = end_char
                    else:
                        current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                        predictions.append(current_entity)
                        current_entity = None

        if current_entity is not None:
            current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
            predictions.append(current_entity)
            
        results.append({"id": id, "predictions": predictions})


    return {"results": results}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
