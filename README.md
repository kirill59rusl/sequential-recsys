# sequential-recsys

Проект по построению **sequential recommender system** (модели рекомендаций, учитывающие последовательность действий пользователя) на датасете **RetailRocket**.

## Датасет

Используется публичный датасет [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — логи взаимодействий пользователей интернет-магазина (просмотры, добавления в корзину, покупки), а также дерево категорий и свойства товаров.

Исходные файлы располагаются в `dataset/raw/`:

| Файл | Описание |
|---|---|
| `events.csv` | события пользователей (view / addtocart / transaction) |
| `category_tree.csv` | иерархия категорий товаров |
| `item_properties_part1.csv`, `item_properties_part2.csv` | свойства товаров (могут изменяться во времени) |

### Скачивание

Сначала склонируйте проект.

Датасет лежит только на Kaggle и требует авторизации через Kaggle API. Настройка(после клонирования проекта):
 
1. Зарегистрируйтесь на [kaggle.com](https://www.kaggle.com), перейдите в `Account -> Settings -> API -> Create New Token`.
2. Скопируйте `.env.example` в `.env` и впишите туда эти значения:
```bash
   cp .env.example .env
```
```
   KAGGLE_USERNAME=ваш_логин
   KAGGLE_KEY=ваш_ключ
```
   `.env` уже в `.gitignore` и никогда не попадёт в git.
3. Примите правила датасета на его [странице на Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (без этого API вернёт 403 даже с валидным токеном).
 
После этого структура папок и сам датасет создаются одной командой:
 
```bash
uv run python -m src.data.download
```
 
Скрипт создаёт `dataset/{raw,processed,artifact}` и скачивает 4 CSV-файла в `dataset/raw/`. Повторный запуск не перекачивает файлы, если они уже на месте (флаг `--force` — перекачать, `--skip-download` — только создать папки).
 

## Структура проекта

```
sequential-recsys/
├── conf/                  # Hydra-конфигурации (эксперименты, модели, датасеты)
├── dataset/
│   ├── raw/                # исходные CSV-файлы RetailRocket
│   ├── processed/          # обработанные/подготовленные данные
│   └── artifact/           # артефакты препроцессинга (энкодеры, словари и т.п.)
├── outputs/                # результаты запусков (логи, чекпоинты, метрики)
├── src/
│   ├── data/
│   │   ├── eda.py              # разведочный анализ (EDA) сырых событий
│   │   ├── preprocess.py       # k-core фильтрация, сессионизация, кодирование id/событий -> full_data.parquet
│   │   ├── sequence.py         # сборка последовательностей по пользователям -> sequences.parquet
│   │   └── seqdataset.py       # torch Dataset/collate_fn для последовательностей
│   ├── models/
│   │   └── sasrec.py           # архитектура SASRec (self-attention)
│   ├── training/
│   │   ├── popbaseline.py            # бейзлайн: top-k популярных товаров
│   │   ├── repeat_baseline.py        # бейзлайн: повтор последнего item + топ-популярные
│   │   ├── weighted_history_base.py  # бейзлайн: взвешенная история (view/addtocart/transaction)
│   │   └── sasrec/
│   │       ├── engine.py           # train_epoch / evaluate для SASRec
│   │       └── train.py            # основной скрипт обучения SASRec (Hydra)
│   └── utils/
│       └── metrics.py          # Hitrate@K, MRR@K, NDCG@K
├── main.py                # точка входа (заглушка)
├── pyproject.toml         # зависимости проекта
└── uv.lock                # зафиксированные версии зависимостей (uv)
```

## Стек технологий

- **Python** ≥ 3.13
- **PyTorch** — обучение моделей
- **Hydra** — управление конфигурациями экспериментов
- **Polars** — обработка табличных данных
- **Pydantic** — валидация конфигов/структур данных
- **Weights & Biases (wandb)** — логирование экспериментов
- **Matplotlib** — визуализация
- **uv** — управление окружением и зависимостями
- **pytest**, **ruff** — тестирование и линтинг (dev-зависимости)

## Установка

Проект использует [uv](https://docs.astral.sh/uv/) для управления зависимостями.

```bash
git clone https://github.com/kirill59rusl/sequential-recsys.git
cd sequential-recsys
uv sync
```

Затем скачивайте датасет(выше).

## Запуск

> Скрипты внутри `src/` импортируют друг друга как пакет (`from src.data.seqdataset import ...`), поэтому их нужно запускать **как модули из корня проекта** через `python -m`, а не напрямую (`python src/data/preprocess.py` работать не будет — Python не найдёт пакет `src`).

Все команды ниже выполняются из корня репозитория. С `uv` — через `uv run python -m ...`, либо активировав окружение (`uv sync`) и используя обычный `python -m ...`.

### 1. EDA (опционально)

Разведочный анализ сырых событий (`dataset/raw/events.csv`): статистика по пользователям/товарам, конверсии, сессии.

```bash
uv run python -m src.data.eda
```

### 2. Препроцессинг

K-core фильтрация, сессионизация, кодирование id и событий. Читает `dataset/raw/events.csv`, пишет `dataset/processed/full_data.parquet` (+ энкодеры в `dataset/artifact/`).

```bash
uv run python -m src.data.preprocess
```

### 3. Сборка последовательностей

Группирует события по пользователям в последовательности. Читает `dataset/processed/full_data.parquet`, пишет `dataset/processed/sequences.parquet`.

```bash
uv run python -m src.data.sequence
```

### 4. Бейзлайны

Оценка простых бейзлайнов (Hitrate@K / MRR@K / NDCG@K) на `dataset/processed/sequences.parquet` и `full_data.parquet`:

```bash
uv run python -m src.training.popbaseline           # top-k популярных товаров
uv run python -m src.training.repeat_baseline        # повтор последнего item + топ-популярные
uv run python -m src.training.weighted_history_base  # взвешенная история (view/addtocart/transaction)
```

### 5. Обучение SASRec

Обучение self-attention модели SASRec с логированием в W&B и early stopping. Конфигурация — через Hydra (`conf/`).

```bash
uv run python -m src.training.sasrec.train
```

Переопределение параметров конфига из командной строки:

```bash
uv run python -m src.training.sasrec.train train.lr=0.001 model.hidden_dim=128
```

Лучший чекпоинт сохраняется в папке outputs/{текущий запуск}/`cfg.train.ckpt_dir`, после обучения модель автоматически прогоняется на тестовой выборке.
