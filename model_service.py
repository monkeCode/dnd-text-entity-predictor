from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import torch
from model import NERModel
from transformers import AutoTokenizer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить запросы с любого источника
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (GET, POST, OPTIONS и т.д.)
    allow_headers=["*"],  # Разрешить все заголовки
)

# Загрузка модели и токенизатора
MODEL_PATH = 'models/model_rubert-tiny2_50_eps.pth'
BASE_MODEL_NAME =  "cointegrated/rubert-tiny2"
NUM_LABELS = 7

with open('data/ner_dataset/labels.txt', 'r') as f:
        label_list = [line.strip() for line in f]

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}


model:NERModel = NERModel(BASE_MODEL_NAME, NUM_LABELS, id2label, label2id, 0, use_prev_label=True)
model.load_state_dict(state_dict=torch.load(MODEL_PATH))
model.eval()


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

class Item(BaseModel):
    batch: List[Dict[str, str]]

@torch.no_grad
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_offsets_mapping=True)
    logits = model(input_ids=torch.tensor(inputs["input_ids"]), attention_mask=torch.tensor(inputs["attention_mask"]), labels=None)["logits"].squeeze(0)
    preds = torch.argmax(logits, dim=-1).tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].tolist()[0])
    offsets = inputs["offset_mapping"].tolist()[0]
    tokens_with_preds = list(zip(tokens, preds, offsets))

    return tokens_with_preds

@app.post('/predict')
def predict_batch(item: Item):
    results = []
    for request in item.batch:
        text = request["text"]
        id = request["id"]
        tokens_with_preds = predict(text)

        # Преобразуем предсказания в формат [начало, конец, класс]
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
                        "class": model.hparams.id2label[pred],
                        "start": start_char,
                        "end": end_char
                    }
                elif current_entity is not None:
                    current_entity["end"] = end_char
            else:
                if current_entity is not None:
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    predictions.append(current_entity)
                    current_entity = None

        if current_entity is not None:
            predictions.append(current_entity)
            
        results.append({"id": id, "predictions": predictions})


    return {"results": results}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
