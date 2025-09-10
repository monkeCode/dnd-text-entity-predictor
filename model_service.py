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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

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

@torch.no_grad()
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, return_offsets_mapping=True)
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=None)["logits"].squeeze(0)
    preds = torch.argmax(logits, dim=-1).tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].tolist()[0])
    offsets = inputs["offset_mapping"].tolist()[0]
    tokens_with_preds = list(zip(tokens, preds, offsets))

    return tokens_with_preds

def exist_next(preds:list, label:str, index):
    for i in range(index, len(preds)):
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
        id = request["id"]
        tokens_with_preds = predict(text)

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
