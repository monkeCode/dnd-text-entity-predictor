# NER for Dungeons & Dragons Entities

End-to-end пайплайн для сбора данных, обучения и развертывания модели для распознавания именованных сущностей (NER) в текстах по вселенной Dungeons & Dragons.

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)

## 📋 Состояние проекта

- [x] Пайплайн сбора и предобработки данных
- [x] Обучение и валидация моделей
- [x] Деплой модели как веб-сервиса (FastAPI)
- [ ] Контейнеризация (Docker)
- [ ] Настройка CI/CD (GitHub Actions)

## 🚀 Мотивация

В настольной ролевой игре Dungeons and Dragons (D&D) правила рассредоточены по многочисленным книгам, и ключевые сущности — **Заклинания**, **Монстры** и **Магические предметы** — часто ссылаются друг на друга. Существующие инструменты NER общего назначения плохо справляются с задачей точного определения и связывания этих узкотематических сущностей.

Данный проект решает проблему автоматического извлечения и классификации D&D-сущностей из текста для последующего создания связанной базы знаний или быстрой навигации по правилам.

## 🗃️ Данные

Данные были получены путем парсинга веб-страниц ресурсов [dnd.su](https://dnd.su/) и [pathfinder.family](https://pathfinder.family/), где сущности уже размечены гиперссылками.

**Статистика датасета:**

- **Общий объем:** 20116 текстовых примеров
- **Общее количество упоминаний сущностей:** 43248
- **Распределение по классам:**
  - Заклинание (Spell): `32315`
  - Монстр (Monster): `9473`
  - Магический предмет (Item): `1460`
- **Разбиение:** Train/Validation/Test = 80%/10%/10%

**Аугментация:** Для повышения устойчивости модели исходные тексты ссылок (`"Красный дракон [Red Dragon]"`) были аугментированы вариантами без переводов в скобках (например, `"Красный дракон"`).

**Пример размеченного текста:**

```json
{
  "text": "Игрок может использовать Заклинание Волшебная стрела, чтобы поразить Кобольда [Cobold], который держит Посох паука.",
  "entities": [
    { "start": 32, "end": 47, "label": "SPELL" },
    { "start": 67, "end": 75, "label": "MONSTER" },
    { "start": 90, "end": 103, "label": "ITEM" }
  ]
}
```

## 🧠 Модель

В основе решения — дообученная языковая модель **BERT** (`cointegrated/rubert-tiny2`), над которой надстроен классификатор токенов.

Было исследовано две архитектуры классификатора:

1. **Авторегрессионная:** Учитывает эмбеддинг текущего токена и предсказанный класс предыдущего токена.
2. **Неавторегрессионная:** Классифицирует токены исключительно на основе их контекстуальных эмбеддингов от BERT.

**Финальная архитектура:** Модель с указанием предыдущей метки класса показала лучшие результаты.

**Гиперпараметры обучения:**

| Параметр                       | Значение                    |
| :----------------------------- | :-------------------------- |
| **Базовая модель**             | `cointegrated/rubert-tiny2` |
| **Learning Rate**              | 1e-5                        |
| **Batch Size**                 | 32                          |
| **Epochs**                     | 50                          |
| **Dropout Rate**               | 0.1                         |
| **Hidden Dim (классификатор)** | 32                          |
| **Оптимизатор**                | AdamW                       |

## 📊 Результаты и метрики

Модель оценивалась на тестовой выборке по метрикам **Precision, Recall и F1-score** (macro-average).

### Общие метрики

| Модель                                     |    Precision    |     Recall      |    F1-score     |
| :----------------------------------------- | :-------------: | :-------------: | :-------------: |
| `rubert-tiny2` + Baseline Classifier       | `{TODO: value}` | `{TODO: value}` | `{TODO: value}` |
| `rubert-tiny2` + Autoregressive Classifier | `{TODO: value}` | `{TODO: value}` | `{TODO: value}` |

### Детальная разбивка по классам (лучшая модель)

| Класс          |      Precision      |       Recall        |      F1-score       |     Support     |
| :------------- | :-----------------: | :-----------------: | :-----------------: | :-------------: |
| **B-SPELL**      |   `{TODO: value}`   |   `{TODO: value}`   |   `{TODO: value}`   | `{TODO: value}` |
| **I-SPELL**      |   `{TODO: value}`   |   `{TODO: value}`   |   `{TODO: value}`   | `{TODO: value}` |
| **B-MONSTER**    |   `{TODO: value}`   |   `{TODO: value}`   |   `{TODO: value}`   | `{TODO: value}` |
| **I-MONSTER**    |   `{TODO: value}`   |   `{TODO: value}`   |   `{TODO: value}`   | `{TODO: value}` |
| **B-ITEM** |   `{TODO: value}`   |   `{TODO: value}`   |   `{TODO: value}`   | `{TODO: value}` |
| **I-ITEM** |   `{TODO: value}`   |   `{TODO: value}`   |   `{TODO: value}`   | `{TODO: value}` |

## 🚀 Деплой

Модель обернута в REST API сервис на основе **FastAPI**.

**Запуск сервиса:**

```bash
python model_service.py
```

**Эндпоинт `POST /predict`**

- **Запрос:**

```json
  {
    "batch": [
      { "id": 0, "text": "Воин атакует двуручным мечом." },
      {
        "id": 1,
        "text": "Маг использует Заговор [Огненный шар] и призывает Скелета."
      }
    ]
  }
  ```

- **Ответ:**

```json
  {
    "results": [
      {
        "id": 0,
        "text": "Воин атакует двуручным мечом.",
        "entities": []
      },
      {
        "id": 1,
        "text": "Маг использует Заговор [Огненный шар] и призывает Скелета.",
        "entities": [
          { "start": 24, "end": 38, "label": "SPELL" },
          { "start": 56, "end": 63, "label": "MONSTER" }
        ]
      }
    ]
  }
  ```

## 🛠️ Установка и запуск

1. **Клонируйте репозиторий:**

    ```bash
    git clone https://github.com/monkeCode/dnd-text-entity-predictor.git
    cd dnd-text-entity-predictor
    ```

2. **Установите зависимости:**

    ```bash
    pip install -r requirements.txt
    ```

3. **(Опционально) Запустите обучение:**

    ```bash
    python train.py --config configs/train_config.yml
    ```

4. **Запустите сервис:**

    ```bash

    ```

    Документация API будет доступна по адресу: `http://localhost:8000/docs`
