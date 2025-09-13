import pandas as pd
import re
from typing import List, Tuple, Optional
import argparse
from pathlib import Path

def parse_annotated_text(text: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Парсит текст с аннотациями в формате [название](ТИП) и возвращает очищенный текст
    и массив аннотаций с позициями и классами.
    
    Args:
        text: строка с аннотациями в формате [название](ТИП)
    
    Returns:
        Tuple[str, List[Tuple[int, int, str]]]: 
            - очищенный текст без аннотаций
            - список аннотаций (start_pos, end_pos, annotation_class)
    """
    # Регулярное выражение для поиска аннотаций [текст](КЛАСС)
    pattern = r'\[(.*?)\]\((.*?)\)'
    
    cleaned_text = ""
    annotations = []
    current_pos = 0
    
    # Ищем все совпадения
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return text, []
    
    # Обрабатываем текст между совпадениями
    last_end = 0
    for match in matches:
        # Текст до аннотации
        text_before = text[last_end:match.start()]
        cleaned_text += text_before
        current_pos += len(text_before)
        
        # Текст внутри аннотации (название объекта)
        object_text = match.group(1)
        annotation_class = match.group(2)
        
        # Добавляем текст объекта в очищенный текст
        start_pos = current_pos
        cleaned_text += object_text
        end_pos = current_pos + len(object_text)
        current_pos = end_pos
        
        # Добавляем аннотацию
        annotations.append((start_pos, end_pos, annotation_class))
        
        last_end = match.end()
    
    # Добавляем оставшийся текст после последней аннотации
    text_after = text[last_end:]
    cleaned_text += text_after
    
    return cleaned_text, annotations

def process_csv_files(input_files: List[str], output_texts_csv: str, output_annotations_csv: str):
    """
    Обрабатывает CSV файлы, извлекает аннотации и сохраняет результаты.
    
    Args:
        input_files: список путей к CSV файлам
        output_texts_csv: путь для сохранения CSV с текстами
        output_annotations_csv: путь для сохранения CSV с аннотациями
    """
    all_texts_data = []
    all_annotations_data = []
    text_id_counter = 1
    
    for file_path in input_files:
        try:
            # Читаем CSV файл
            df = pd.read_csv(file_path)
            print(f"Обрабатываю файл: {file_path} ({len(df)} строк)")
            
            # Обрабатываем каждую строку
            for _, row in df.iterrows():
                # Получаем предложения из строки
                if 'sentense' in row:
                    sentence = row['sentense']
                    
                    # Парсим аннотации
                    cleaned_text, annotations = parse_annotated_text(sentence)
                    
                    # Добавляем текст в коллекцию
                    text_data = {
                        'id': str(text_id_counter) + "_s",
                        'text': cleaned_text,
                        'source': file_path
                    }
                    all_texts_data.append(text_data)
                    
                    # Добавляем аннотации
                    for start_pos, end_pos, annotation_class in annotations:
                        if annotation_class not in ["MONSTER", "SPELL", "ITEM"]:
                            continue
                        
                        annotation_data = {
                            'text_id': str(text_id_counter) + "_s",
                            'start': start_pos,
                            'end': end_pos,
                            'class': annotation_class,
                        }
                        all_annotations_data.append(annotation_data)
                    
                    text_id_counter += 1
                        
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")
            continue
    
    # Создаем DataFrame для текстов
    texts_df = pd.DataFrame(all_texts_data)
    
    # Создаем DataFrame для аннотаций
    annotations_df = pd.DataFrame(all_annotations_data)
    
    # Сохраняем результаты
    if not texts_df.empty:
        texts_df.to_csv(output_texts_csv, index=False, encoding='utf-8')
        print(f"Сохранено {len(texts_df)} текстов в {output_texts_csv}")
    else:
        print("Нет данных для сохранения в texts CSV")
    
    if not annotations_df.empty:
        annotations_df.to_csv(output_annotations_csv, index=False, encoding='utf-8')
        print(f"Сохранено {len(annotations_df)} аннотаций в {output_annotations_csv}")
    else:
        print("Нет данных для сохранения в annotations CSV")

def main():
    parser = argparse.ArgumentParser(description='Обработка CSV файлов с аннотированными текстами D&D')
    parser.add_argument('--input_files', nargs='+', required=True, help='Список CSV файлов для обработки')
    parser.add_argument('--output_texts', required=True, help='Путь для сохранения CSV с текстами')
    parser.add_argument('--output_annotations', required=True, help='Путь для сохранения CSV с аннотациями')
    
    args = parser.parse_args()
    
    # Проверяем существование входных файлов
    missing_files = [f for f in args.input_files if not Path(f).exists()]
    if missing_files:
        print(f"Ошибка: следующие файлы не найдены: {missing_files}")
        return
    
    # Обрабатываем файлы
    process_csv_files(args.input_files, args.output_texts, args.output_annotations)

if __name__ == "__main__":
    # Пример использования из командной строки:
    # python script.py --input_files file1.csv file2.csv --output_texts texts_synt.csv --output_annotations annotations_synt.csv
    main()