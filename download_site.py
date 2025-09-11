import requests
import bs4
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
import time

# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description='Download pages from dnd.su')
parser.add_argument('--output', '-o', type=str, default='data/pages.csv',
                    help='Output CSV file path (default: data/pages.csv)')
parser.add_argument('--workers', '-w', type=int, default=5,
                    help='Number of parallel workers (default: 5)')
parser.add_argument('--retries', '-r', type=int, default=3,
                    help='Number of retries for failed downloads (default: 3)')
parser.add_argument('--delay', '-d', type=float, default=1.0,
                    help='Delay between retries in seconds (default: 1.0)')
args = parser.parse_args()

# Создание директории для выходного файла если нужно
os.makedirs(os.path.dirname(args.output), exist_ok=True)

sources = ["https://dnd.su/bestiary/", "https://dnd.su/items/", "https://dnd.su/spells/", "https://dnd.su/homebrew/items/", "https://dnd.su/homebrew/spells/", "https://dnd.su/homebrew/bestiary/"]
refs = []

for s in sources:
    page = requests.get(s).text
    soup = bs4.BeautifulSoup(page, 'html.parser')
    refs.extend([
        "https://dnd.su" + el.get("href") 
        for el in soup.find_all("a") 
        if el.get("href") and any(
            el.get("href").startswith(path) 
            for path in ["/bestiary", "/items", "/spells", "/homebrew/spells", "/homebrew/bestiary", "/homebrew/items"]
        )
    ])

refs = list(set(refs))

def download_page(url, max_retries=3, delay=1.0):
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

# Создаем DataFrame со всеми ссылками
frame = pd.DataFrame(refs, columns=["href"])

# Параллельная загрузка с прогресс-баром
results = []
with ThreadPoolExecutor(max_workers=args.workers) as executor:
    # Запускаем загрузку
    future_to_url = {
        executor.submit(
            download_page, 
            row["href"],
            max_retries=args.retries,
            delay=args.delay
        ): row["href"] 
        for _, row in frame.iterrows()
    }
    
    # Обрабатываем результаты с прогресс-баром
    for future in tqdm(
        as_completed(future_to_url), 
        total=len(future_to_url),
        desc="Downloading pages"
    ):
        url = future_to_url[future]
        try:
            page_content, status_code = future.result()
            results.append({
                "href": url,
                "page": page_content,
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