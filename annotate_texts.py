from typing import Any
import requests
import bs4
import pandas as pd
import argparse
import os
import csv
from tqdm import tqdm
import re
import json

# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description='NER markup for dnd.su pages')
parser.add_argument('--input', '-i', type=str, default='data/pages.csv',
                    help='Input CSV file with pages (default: data/pages.csv)')

parser.add_argument('--input_pathfinder', type=str, default='data/pathfinder_json.csv',
                    help='Input CSV file with jsons (default: data/pathfinder_json.csv)')

parser.add_argument('--output-texts', '-ot', type=str, default='data/texts.csv',
                    help='Output CSV file for texts (default: data/texts.csv)')
parser.add_argument('--output-annotations', '-oa', type=str, default='data/annotations.csv',
                    help='Output CSV file for annotations (default: data/annotations.csv)')
args = parser.parse_args()

# Создание директорий для выходных файлов если нужно
os.makedirs(os.path.dirname(args.output_texts), exist_ok=True)
os.makedirs(os.path.dirname(args.output_annotations), exist_ok=True)

def decompose_tag(text, sub_page: bs4.BeautifulSoup | bs4.PageElement, start_idx=0):
    i = start_idx
    links = []
    if type(sub_page) is bs4.element.NavigableString:
        return text + sub_page.text + " ", []
    
    for el in sub_page.contents:
        if el.name == "a":
            s = i
            t = el.text
            e = i + len(t)
            text += t + " "
            href = el.get("href", "")
            if href.startswith("/spell"):
                clas = "SPELL"
            elif href.startswith("/bestiary"):
                clas = "MONSTER"
            elif href.startswith("/items") or href.startswith("/magicItem"):
                clas = "ITEM"
            else:
                continue

            links.append((s, e, clas))
        else:
            text, l = decompose_tag(text, el, i)
            links.extend(l)
        i = len(text)

    return text, links

def parse_page(page_html) -> tuple[list[Any], list[Any]]:
    texts = []
    annots = []
    try:
        page = bs4.BeautifulSoup(page_html, 'html.parser')
        content: bs4.Tag = page.find(class_="card__body new-article")
        if not content:
            return texts, annots

        for p in [ it for d in content.find_all(class_="subsection desc") for it in (d.find_all("p") + d.find_all("ul") + (d.find_all("td")))]:
            if len(p.text) < 10: continue
            text, annotations = decompose_tag("", p)
            texts.append(text)
            annots.append(annotations)

        return texts, annots
    except Exception as e:
        print(f"Error parsing page: {e}")
        return texts, annots

def parse_pathfinder_page(page_html):
    try:
        page = bs4.BeautifulSoup(page_html, "html.parser")
        # text = page.text
        text, annotations = decompose_tag("", page)
        return [text], [annotations]
    except Exception as e:
        print(f"Error parsing page: {e}")
        return [], []

def parse_json(js):
    texts = []
    for k, v in js.items():
        if type(v) is not str or v is None:
            continue
        if len(v) > 100:
            texts.append("<span>" + v + "</span>")

    return texts

def augment_text(text, annotations):
    # Функция для обработки текста внутри аннотации
    if len(annotations) == 0:
        return [(text, annotations)]

    def process_segment(segment_text, segment_start):
        # Используем регулярное выражение для поиска текста в квадратных скобках
        pattern = re.compile(r'\[([^]]+)\]')

        # Находим все совпадения
        matches = list(pattern.finditer(segment_text))

        if not matches:
            return segment_text, 0  # Возвращаем текст и сдвиг, если нет изменений

        # Удаляем текст в квадратных скобках
        new_text = pattern.sub('', segment_text)

        # Вычисляем общий сдвиг для обновления позиций аннотаций
        shift = 0
        for match in matches:
            shift += (match.end() - match.start())

        return new_text, shift

    # Основной текст и аннотации
    new_text = text
    new_annotations = list(sorted(annotations, key=lambda x: x[0]))
    shift = 0
    # Обрабатываем каждую аннотацию
    for ann_idx, (ann_start, ann_end, ann_label) in enumerate(annotations):
        segment_text = text[ann_start:ann_end]

        # Обрабатываем сегмент
        processed_text, s = process_segment(segment_text, ann_start)
        # Обновляем основной текст
        new_text = new_text[:ann_start-shift] + processed_text + new_text[ann_end-shift:]

        # Обновляем аннотации, учитывая смещение из-за удаления текста
        # Корректируем текущую аннотацию
        new_annotations[ann_idx] = (ann_start-shift, ann_start + len(processed_text)-shift, ann_label)

        shift += s

    if(new_text == text):
        return [(text,annotations)]

    return [(new_text, new_annotations), (text, annotations)]

pathfinder_frame = pd.read_csv(args.input_pathfinder)

pathfinder_pages = pd.DataFrame([{"href": i["href"], "page":text, "code":200 } for _, i in pathfinder_frame.iterrows() if i["code"] == "200" for text in parse_json(json.loads(i["page"]))])

# Чтение входных данных
print(f"Reading input from {args.input}")
df = pd.concat([pd.read_csv(args.input),pathfinder_pages], axis=0)
df.reset_index(inplace=True)

# Обработка страниц
texts_data = []
annotations_data = []

print("Processing pages...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    href = row['href']
    page_html = row['page']
    status_code = row['code']
    
    # Пропускаем страницы с ошибками
    if status_code != 200 or not isinstance(page_html, str):
        continue
    
    # Парсим страницу
    if "pathfinder.family" in href:
        texts, annotations_list = parse_pathfinder_page(page_html)
    else:
        texts, annotations_list = parse_page(page_html)
    for t_i, (text, annotations) in enumerate(zip(texts, annotations_list)):
        augmented_variants = augment_text(text, annotations)
        for variant_id, (variant_text, variant_annotations) in enumerate(augmented_variants):
            new_id = f"{idx}_{t_i}_{variant_id}"
            texts_data.append({
                'id': new_id,
                'href': href,
                'text': variant_text,
                'variant': variant_id,
                "unique_id": f"{idx}_{t_i}"
            })
            
            for ann in variant_annotations:
                annotations_data.append({
                    'text_id': new_id,
                    'start': ann[0],
                    'end': ann[1],
                    'class': ann[2]
                })

# Сохранение результатов
print(f"Saving texts to {args.output_texts}")
texts_df = pd.DataFrame(texts_data)
texts_df.to_csv(args.output_texts, index=False, quoting=csv.QUOTE_ALL)

print(f"Saving annotations to {args.output_annotations}")
annotations_df = pd.DataFrame(annotations_data)
annotations_df.to_csv(args.output_annotations, index=False, quoting=csv.QUOTE_ALL)

print(f"Processed {len(texts_data)} pages with {len(annotations_data)} annotations")