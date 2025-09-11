import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
import torch
from tqdm import tqdm
import argparse
import os
from ner_dataset import NERDataset
import dvc.api

params = dvc.api.params_show()
# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description='Prepare NER dataset from annotations')
parser.add_argument('--texts', type=str, required=True, help='Path to texts CSV file')
parser.add_argument('--annotations', type=str, required=True, help='Path to annotations CSV file')
# parser.add_argument('--model_name', type=str, default='ai-forever/ruBert-base', 
#                     help='HuggingFace model name (default: ai-forever/ruBert-base)')
parser.add_argument('--output_dir', type=str, default='data/ner_dataset',
                    help='Output directory for dataset (default: data/ner_dataset)')
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='Ratio of data for training (default: 0.8)')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='Ratio of data for validation (default: 0.1)')
parser.add_argument('--test_ratio', type=float, default=0.1,
                    help='Ratio of data for testing (default: 0.1)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42)')
args = parser.parse_args()

# Создание выходной директории
os.makedirs(args.output_dir, exist_ok=True)

# Загрузка данных
print("Loading data...")
texts_df = pd.read_csv(args.texts)
annotations_df = pd.read_csv(args.annotations)

# Создание словаря меток
label_list = ['O', 'B-SPELL', 'I-SPELL', 'B-MONSTER', 'I-MONSTER', 'B-ITEM', 'I-ITEM']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Сохранение меток
with open(os.path.join(args.output_dir, 'labels.txt'), 'w') as f:
    for label in label_list:
        f.write(f"{label}\n")

# Инициализация токенизатора
model_name = params["base-model"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = params["max-tokens"]


# Разделение данных на train/val/test
text_ids = texts_df['unique_id'].unique()
np.random.seed(args.seed)
np.random.shuffle(text_ids)

train_size = int(len(text_ids) * args.train_ratio)
val_size = int(len(text_ids) * args.val_ratio)

train_ids = text_ids[:train_size]
val_ids = text_ids[train_size:train_size + val_size]
test_ids = text_ids[train_size + val_size:]

train_texts = texts_df[texts_df['unique_id'].isin(train_ids)]
val_texts = texts_df[texts_df['unique_id'].isin(val_ids)]
test_texts = texts_df[texts_df['unique_id'].isin(test_ids)]

train_annotations = annotations_df[annotations_df['text_id'].isin(train_ids)]
val_annotations = annotations_df[annotations_df['text_id'].isin(val_ids)]
test_annotations = annotations_df[annotations_df['text_id'].isin(test_ids)]

# Создание датасетов
print("Creating datasets...")
train_dataset = NERDataset(train_texts, train_annotations, tokenizer,max_length , label2id)
val_dataset = NERDataset(val_texts, val_annotations, tokenizer, max_length, label2id)
test_dataset = NERDataset(test_texts, test_annotations, tokenizer, max_length, label2id)

# Сохранение датасетов
print("Saving datasets...")
torch.save(train_dataset, os.path.join(args.output_dir, 'train_dataset.pt'))
torch.save(val_dataset, os.path.join(args.output_dir, 'val_dataset.pt'))
torch.save(test_dataset, os.path.join(args.output_dir, 'test_dataset.pt'))

# Сохранение информации о разделении
split_info = {
    'train_ids': train_ids.tolist(),
    'val_ids': val_ids.tolist(),
    'test_ids': test_ids.tolist()
}

import json
with open(os.path.join(args.output_dir, 'split_info.json'), 'w') as f:
    json.dump(split_info, f)

print(f"Dataset created and saved to {args.output_dir}")
print(f"Train: {len(train_dataset)} examples")
print(f"Validation: {len(val_dataset)} examples")
print(f"Test: {len(test_dataset)} examples")
print(f"Labels: {label_list}")