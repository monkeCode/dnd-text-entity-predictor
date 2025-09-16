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
import numpy as np
from collections import defaultdict

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

SPLIT_BY_SENTENSES = False

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
def predict(text, max_length=200, overlap=0.2):
    """
    Предсказывает NER-метки для текста с использованием скользящего окна.
    Сначала токенизирует весь текст, затем обрабатывает части с перекрытием.
    
    Args:
        text (str): Входной текст для анализа
        max_length (int): Максимальная длина последовательности для модели
        overlap (float): Процент перекрытия между окнами (0.0-1.0)
    
    Returns:
        list: Список кортежей (токен, предсказание, смещение)
    """
    # Проверяем, что перекрытие в допустимом диапазоне
    overlap = max(0.0, min(0.9, overlap))
    
    full_encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False
    )
    
    all_tokens = full_encoding["input_ids"]
    all_offsets = full_encoding["offset_mapping"]
    
    num_tokens = len(all_tokens)
    window_size = max_length - 2 
    stride = int(window_size * (1 - overlap))
    
    token_logits = {i:[] for i in range(num_tokens)}
    
    for start_idx in range(0, num_tokens, stride):
        end_idx = min(start_idx + window_size, num_tokens)
        
        inputs = tokenizer.prepare_for_model(all_tokens[start_idx:end_idx], return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_offsets_mapping=True)
        
        outputs = model(
            input_ids=inputs["input_ids"].to(model.device).unsqueeze(0),
            attention_mask=inputs["attention_mask"].to(model.device).unsqueeze(0)
        )
        
        chunk_logits = outputs["logits"].squeeze(0)
        
        chunk_logits = [chunk_logits[i] for i in range(len(chunk_logits)) if inputs["input_ids"][i].item() not in tokenizer.all_special_ids ]

        for i,j in enumerate(range(start_idx, end_idx)):
            token_logits[j].append(chunk_logits[i])
    
    result = []
    for token_idx in range(num_tokens):
        if token_idx not in token_logits:
            print("ERROR - TOKEN NOT IN DICT")
            continue
            
        avg_logits = np.mean(token_logits[token_idx], axis=0)
        pred = np.argmax(avg_logits)
        
        token_text = text[all_offsets[token_idx][0]:all_offsets[token_idx][1]]
        
        result.append((
            token_text,
            int(pred),
            all_offsets[token_idx]
        ))
    
    return result

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
            sentens = re.split(r'(?<=[\n.])(?=\S)', text)
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

    

    print(results)
    return {"results": results}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
