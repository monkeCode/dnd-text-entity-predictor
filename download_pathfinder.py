import requests
import bs4
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
import time
import json

# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description='Download json from pathfinder.family')
parser.add_argument('--output', '-o', type=str, default='data/pathfinder_json.csv',
                    help='Output CSV file path (default: data/pathfinder_json.csv)')
parser.add_argument('--workers', '-w', type=int, default=3,
                    help='Number of parallel workers (default: 3)')
parser.add_argument('--retries', '-r', type=int, default=3,
                    help='Number of retries for failed downloads (default: 3)')
parser.add_argument('--delay', '-d', type=float, default=1.0,
                    help='Delay between retries in seconds (default: 1.0)')
args = parser.parse_args()

# Создание директории для выходного файла если нужно
os.makedirs(os.path.dirname(args.output), exist_ok=True)

monsters = requests.get("http://pathfinder.family/api/beasts").json(), 
spells = requests.get("http://pathfinder.family/api/spells").json(), 
items = requests.get("http://pathfinder.family/api/allMagicItems").json()

if len(monsters) == 1:
    monsters = monsters[0]

if len(spells) == 1:
    spells = spells[0]

frame_hrefs = pd.DataFrame([("http://pathfinder.family//api/beastInfo?alias="+m["alias"], "monster") for m in monsters] + 
          [("http://pathfinder.family//api/spellInfo?alias=" + s["alias"], "spell") for s in spells ]+ 
          [("http://pathfinder.family//api/magicItemInfo?alias=" + i["alias"], "item") for i in items], columns=["href", "type"])



def get_json(url, max_retries=3, delay=1.0):
    """Загружает страницу с повторными попытками"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()  # Проверка статуса
            return resp.text, resp.status_code
        except (requests.RequestException, ConnectionError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return None, str(e)
    return None, "Max retries exceeded"


# Параллельная загрузка с прогресс-баром
results = []
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    # Запускаем загрузку
    future_to_url = {
        executor.submit(
            get_json, 
            row["href"],
            max_retries=args.retries,
            delay=args.delay
        ): (row["href"], row["type"])
        for _, row in frame_hrefs.iterrows()
    }
    
    # Обрабатываем результаты с прогресс-баром
    for future in tqdm(
        as_completed(future_to_url), 
        total=len(future_to_url),
        desc="Downloading pages"
    ):
        url = future_to_url[future]
        try:
            json_page, status_code = future.result()
            results.append({
                "href": url[0],
                "type": url[1],
                "page": json_page,
                "code": status_code
            })
        except Exception as e:
            results.append({
                "href": url,
                "page": None,
                "code": str(e)
            })

# Создаем итоговый DataFrame и сохраняем
res_frame = pd.DataFrame(results)
res_frame.to_csv(args.output, index=False)
print(f"Saved {len(results)} pages to {args.output}")