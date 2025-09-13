import requests
import pandas as pd
import json
import csv
import time
from pathlib import Path
import numpy as np


# Настройки LM Studio
LMSTUDIO_HOST = "http://localhost:1234"
ENDPOINT = "/v1/chat/completions"


OUTPUT_FILE = "data/lmstudio_responses.csv"
BATCH_SIZE = 5

# Промпт для запроса
PROMPT = """ты модель генератор текстов по днд, тебе необходимо сгенерировать 5 параграфов текстов в днд тематике с присутствием монствов, заклинаний или магических предметов в тексте. каждый такой объект помечай как ссылку в makrdown и в круглых скобках пиши, что это за объект (SPELL, MONSTER, ITEM)

ПРИМЕР ОФОМЛЕНИЯ ОБЪЕКТОВ:
[Волшебная стрела](SPELL)
[Высший вампир](MOSTER)
[Чудесные краски](ITEM)

ответь в формате json по следующему примеру :
```json
{
"reasoning": "Я использую такие заклинания, как обнаружение зла и добра, обнаружении магии и исполнение желаний, монстров летучих мышей, волков и рой крыс, а также магические предметы: универсальный растворитель и масло эфирности, нужно сгенерировать 5 предложений и сделать их как можно более разнообразными по составу объектов, но при этом логичными и последовательными"
"sentenses": [ " В любое время во время вашего хода [сфера](ITEM) может наложить заклинание  [внушение](SPELL)  (Сл спасброска 17), нацеленное на вас или другое существо, которое коснулось шара в течение последних 24 часов. Это [сила сферы](ITEM), которую вы не контролируете.", 
"Неограниченно:  [изменение формы камня](SPELL)  ,  [обнаружение зла и добра](SPELL)  ,  [обнаружение магии](SPELL)",
"Например, если вы жрец 3-го уровня, то у вас есть четыре ячейки заклинаний 1-го уровня и две ячейки 2-го уровня. При Мудрости 16 ваш список подготовленных заклинаний может включать в себя шесть заклинаний 1-го или 2-го уровня в любой комбинации. Если вы подготовили заклинание 1-го уровня [лечение ран](SPELL), вы можете наложить его, используя ячейку 1-го уровня или ячейку 2-го уровня. Накладывание заклинания не удаляет его из списка подготовленных заклинаний.",
"Дети ночи (1/день). Вампир магическим образом призывает 2к4 [роя крыс](MONSTER) или [летучих мышей](MONSTER, при условии, что на небе нет солнца. Находясь на открытом воздухе, вампир может вместо этого призвать 3к6 [волков](MONSTER). Вызванные существа приходят через 1к4 раунда, действуют как союзники вампира и подчиняются его устным командам. Звери остаются на 1 час, пока вампир не умрет, или пока вампир не отпустит их бонусным действием.",
"Эта тягучая, молочно-белая субстанция может склеить два любых предмета. Она должна храниться в сосуде или фляге, покрытой изнутри [маслом скольжения](ITEM). В найденном контейнере находится 1к6 + 1 унция клея. Одна унция клея может покрыть 1 квадратный фут поверхности. Клей затвердевает через 1 минуту. После этого связь двух предметов можно разорвать только нанесением [универсального растворителя](ITEM) или [масла эфирности](ITEM), либо же заклинанием [исполнение желаний](ITEM)."]
}
```

не пиши никаких символов вне этого json, твой ответ обязательно должен быть на русском языке"""


def make_request(objects):
    """Отправляет запрос к LM Studio и возвращает ответ"""

    monsters = np.random.choice(objects[objects.type == "MONSTER"].name, size=3)
    items = np.random.choice(objects[objects.type == "ITEM"].name, size=3)
    spells = np.random.choice(objects[objects.type == "SPELL"].name, size=3)

    dop_text = f'''Используй следующие объекты: 
        монстров: {", ".join([f"[{m}](MONSTER)" for m in monsters])}\n магические предметы: {", ".join([f"[{i}](ITEM)" for i in items])}\nЗаклинания: {", ".join([f"[{s}](SPELL)" for s in spells])}'''


    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "local-model",  # Может потребоваться изменить на актуальное имя модели
        "messages": [
            {"role": "system", "content": "Ты помощник, который генерирует тексты в тематике D&D."},
            {"role": "user", "content": PROMPT + "\n" + dop_text}
        ],
        "temperature": 0.7,
        "max_tokens": 2000,
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{LMSTUDIO_HOST}{ENDPOINT}",
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Ошибка при запросе: {e}")
        return None

def extract_json_from_response(response_text):
    """Извлекает JSON из текста ответа"""
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx + 1].strip()
            return json.loads(json_str)
        else:
            # Пробуем распарсить весь текст как JSON
            return json.loads(response_text)
    except Exception as e:
        print(f"Ошибка при парсинге JSON: {e}")
        return {"error": str(e), "raw_response": response_text}

def save_to_csv(responses, filename):
    file_exists = Path(filename).exists()
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'reasoning', 'sentense']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for response in responses:
            sentenses = response['data'].get('sentenses', [])
            for s in sentenses:
                writer.writerow({
                    'timestamp': response['timestamp'],
                    'reasoning': response['data'].get('reasoning', ''),
                    'sentense': s
                })

def main(frame):
    responses_batch = []
    request_count = 0
    
    print("Скрипт запущен. Для остановки нажмите Ctrl+C")
    
    try:
        while True:
            print(f"Делаю запрос #{request_count + 1}...")
            result = make_request(frame)
            
            if result:
                content = result['choices'][0]['message']['content']
                parsed_response = extract_json_from_response(content)
                
                responses_batch.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'data': parsed_response
                })
                
                print(f"Запрос #{request_count + 1} завершен успешно")
                request_count += 1
                
                if len(responses_batch) >= BATCH_SIZE:
                    print(f"Сохраняю {len(responses_batch)} ответов в CSV...")
                    save_to_csv(responses_batch, OUTPUT_FILE)
                    responses_batch = []
                    print("Данные сохранены")
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\nОстанавливаю скрипт...")
        
        if responses_batch:
            print(f"Сохраняю оставшиеся {len(responses_batch)} ответов...")
            save_to_csv(responses_batch, OUTPUT_FILE)
        
        print(f"Всего выполнено запросов: {request_count}")
        print("Скрипт остановлен")


if __name__ == "__main__":
    frame = pd.read_csv("data/object_list.csv")
    main(frame)