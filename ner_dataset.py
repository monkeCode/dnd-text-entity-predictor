from torch.utils.data import Dataset
from tqdm import tqdm

class NERDataset(Dataset):
    def __init__(self, texts, annotations, tokenizer, max_length, label2id):
        self.texts = texts
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        
        # Предварительная обработка данных
        self.examples = self._preprocess_data()
    
    def _preprocess_data(self):
        examples = []
        
        for text_id, text_row in tqdm(self.texts.iterrows(), total=len(self.texts), desc="Preprocessing data"):
            text = text_row['text']
            text_annotations = self.annotations[self.annotations['text_id'] == text_row['id']]
            
            # Создание меток для каждого символа
            char_labels = ['O'] * len(text)
            for _, ann_row in text_annotations.iterrows():
                start, end, label_class = ann_row['start'], ann_row['end'], ann_row['class']
                
                # Убедимся, что индексы в пределах текста
                start = max(0, min(start, len(text)))
                end = max(0, min(end, len(text)))
                
                if start < end:
                    # Первый токен сущности получает B- префикс
                    char_labels[start] = f'B-{label_class}'
                    
                    # Остальные токены сущности получают I- префикс
                    for i in range(start + 1, end):
                        if i < len(char_labels):
                            char_labels[i] = f'I-{label_class}'
            
            # Токенизация текста и выравнивание меток
            tokenized = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_offsets_mapping=True,
                return_tensors=None
            )
            
            # Получение меток для каждого токена
            labels = []
            offset_mapping = tokenized['offset_mapping']
            
            for i, (start, end) in enumerate(offset_mapping):
                # Специальные токены получают метку -100 (игнорируются в loss function)
                if start == end == 0:
                    labels.append(-100)
                    continue
                
                # Определяем метку для токена на основе меток символов
                token_labels = char_labels[start:end]
                
                # Если все символы в токене имеют метку O, присваиваем O
                if all(label == 'O' for label in token_labels):
                    labels.append(self.label2id['O'])
                else:
                    # Ищем первую не-O метку в токене
                    for label in token_labels:
                        if label != 'O':
                            labels.append(self.label2id[label])
                            break
                    else:
                        labels.append(self.label2id['O'])
            
            examples.append({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': labels
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]