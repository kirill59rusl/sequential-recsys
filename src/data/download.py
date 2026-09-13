"""

    1. Зарегистрируйтесь на https://www.kaggle.com
    2. Account -> Settings -> API -> "Create New Token"
       скачается файл kaggle.json
    3. Положите его в:
         Linux/macOS: ~/.kaggle/kaggle.json
         Windows:     C:\\Users\\<user>\\.kaggle\\kaggle.json
       или задайте переменные окружения KAGGLE_USERNAME / KAGGLE_KEY

Использование:
    uv run python -m src.data.download
    uv run python -m src.data.download --force          # перекачать заново
    uv run python -m src.data.download --skip-download  # только создать папки
"""

import argparse
import shutil
from pathlib import Path
 
from dotenv import load_dotenv
 
load_dotenv()  # подхватывает KAGGLE_USERNAME/KAGGLE_KEY из .env, если он есть
 
DATASET_SLUG = "retailrocket/ecommerce-dataset"
 
DATASET_ROOT = Path("dataset")
SUBDIRS = ["raw", "processed", "artifact"]
 
EXPECTED_FILES = [
    "events.csv",
    "category_tree.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
]
 
 
def create_folder_structure(root: Path = DATASET_ROOT) -> None:
    """Создаёт dataset/{raw,processed,artifact}, если их ещё нет."""
    for sub in SUBDIRS:
        path = root / sub
        path.mkdir(parents=True, exist_ok=True)
        # .gitkeep, чтобы пустые папки не терялись в git
        if not any(path.iterdir()):
            (path / ".gitkeep").touch()
    print(f"Структура создана: {[str(root / s) for s in SUBDIRS]}")
 
 
def already_downloaded(raw_dir: Path) -> bool:
    return all((raw_dir / f).exists() for f in EXPECTED_FILES)
 
 
def download_retailrocket(raw_dir: Path = DATASET_ROOT / "raw", force: bool = False) -> None:
    """Скачивает и распаковывает RetailRocket dataset с Kaggle в raw_dir."""
    if already_downloaded(raw_dir) and not force:
        print("Файлы датасета уже на месте, пропускаю скачивание (--force для перекачки).")
        return
 
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise ImportError(
            "Пакет 'kaggle' не установлен. Добавьте зависимость: uv add kaggle"
        ) from e
 
    api = KaggleApi()
    api.authenticate()  # читает ~/.kaggle/kaggle.json или KAGGLE_USERNAME/KAGGLE_KEY
 
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю {DATASET_SLUG} в {raw_dir} ...")
    api.dataset_download_files(DATASET_SLUG, path=str(raw_dir), unzip=True, quiet=False)
 
    # Kaggle иногда распаковывает во вложенную папку - поднимаем файлы на уровень raw/
    for item in list(raw_dir.iterdir()):
        if item.is_dir():
            for f in item.iterdir():
                shutil.move(str(f), str(raw_dir / f.name))
            item.rmdir()
 
    missing = [f for f in EXPECTED_FILES if not (raw_dir / f).exists()]
    if missing:
        print(f"⚠️  Не найдены ожидаемые файлы после скачивания: {missing}")
    else:
        print("Готово. Файлы в dataset/raw/:")
        for f in EXPECTED_FILES:
            size_mb = (raw_dir / f).stat().st_size / (1024 * 1024)
            print(f"  {f} ({size_mb:.1f} МБ)")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Подготовка структуры dataset/ и загрузка RetailRocket с Kaggle"
    )
    parser.add_argument("--force", action="store_true", help="Перекачать датасет, даже если файлы уже есть")
    parser.add_argument("--skip-download", action="store_true", help="Только создать папки, без скачивания")
    args = parser.parse_args()
 
    create_folder_structure()
 
    if not args.skip_download:
        download_retailrocket(force=args.force)
 
 
if __name__ == "__main__":
    main()
 