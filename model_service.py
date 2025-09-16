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
def predict(text, max_length=200, overlap=0.3):
    """
    Предсказывает NER-метки для текста с использованием скользящего окна
    и агрегацией результатов с перекрытием.
    
    Args:
        text (str): Входной текст для анализа
        max_length (int): Максимальная длина последовательности для модели
        overlap (float): Процент перекрытия между окнами (0.0-1.0)
    
    Returns:
        list: Список кортежей (токен, предсказание, смещение)
    """
    # Проверяем, что перекрытие в допустимом диапазоне
    overlap = max(0.0, min(0.9, overlap))
    
    # Инициализируем массивы для агрегации предсказаний
    all_logits = []
    all_offsets = []
    
    # Определяем размер окна и шаг в символах
    window_size = max_length * 4  # Примерный размер окна в символах
    stride = int(window_size * (1 - overlap))
    
    # Обрабатываем текст окнами
    for start_idx in range(0, len(text), stride):
        end_idx = min(start_idx + window_size, len(text))
        chunk_text = text[start_idx:end_idx]
        
        # Токенизируем чанк с учетом специальных токенов
        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_offsets_mapping=True
        )
        
        # Получаем предсказания для чанка
        outputs = model(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device)
        )
        
        # Корректируем смещения для глобальных позиций
        chunk_offsets = inputs["offset_mapping"].squeeze(0).tolist()
        adjusted_offsets = []
        
        for start, end in chunk_offsets:
            if start == 0 and end == 0:  # Специальные токены
                adjusted_offsets.append((0, 0))
            else:
                adjusted_offsets.append((start + start_idx, end + start_idx))
        
        # Сохраняем логиты и смещения
        all_logits.append(outputs["logits"].squeeze(0))
        all_offsets.extend(adjusted_offsets)
    
    # Объединяем результаты
    if not all_logits:
        return []
    
    combined_logits = torch.cat(all_logits, dim=0)
    preds = torch.argmax(combined_logits, dim=-1).tolist()
    
    # Формируем результат, исключая специальные токены
    result = []
    for i, (start, end) in enumerate(all_offsets):
        # Пропускаем специальные токены и паддинг
        if start == 0 and end == 0:
            continue
            
        # Получаем текст токена
        token_text = text[start:end]
        
        result.append((
            token_text,
            preds[i],
            (start, end)
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


    return {"results": results}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
