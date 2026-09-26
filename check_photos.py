"""
Пакетная проверка фото на наличие завалов
Читает Excel, прогоняет через модель, сохраняет результат
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from tqdm import tqdm
import traceback
import os
import re
# ==================== ЗАГРУЗКА МОДЕЛИ С HUGGINGFACE ====================
from huggingface_hub import hf_hub_download

# ⚠️ ЗАМЕНИ НА СВОЙ HF-РЕПОЗИТОРИЙ
HF_REPO_ID = "AT3MI/trash-detector-yolo11s"
HF_FILENAME = "best.pt"

# Локальная папка и путь
MODEL_DIR = Path("model_weights")
LOCAL_MODEL_PATH = MODEL_DIR / HF_FILENAME

def ensure_model() -> Path:
    """
    Гарантирует, что best.pt есть локально.
    Если нет — создаёт model_weights/ и качает с HuggingFace.
    Возвращает путь к файлу.
    """
    if LOCAL_MODEL_PATH.exists():
        print(f"✅ Модель найдена: {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH

    print(f"⏳ Модель не найдена. Скачиваю с HuggingFace: {HF_REPO_ID}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=str(MODEL_DIR),
    )
    print(f"✅ Модель скачана: {downloaded}")
    return Path(downloaded)

# ======================================================================
# ==================== НАСТРОЙКИ ====================
INPUT_FILE = 'выполнено с фото из путевого.xlsx'
OUTPUT_FILE = f'площадки_с_завалами_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
MODEL_PATH = str(ensure_model())
CONF = 0.3
IMG_SIZE = 640
BATCH_SIZE = 16

# ==================== ФУНКЦИЯ НОРМАЛИЗАЦИИ ПУТИ ====================
def normalize_path(path_str):
    """Приводит путь к правильному UNC-виду: \\server\share\..."""
    if pd.isna(path_str) or not isinstance(path_str, str):
        return None

    # Убираем пробелы по краям
    path_str = path_str.strip()

    # Убираем кавычки если есть
    path_str = path_str.strip('"\'')

    # Заменяем прямые слеши на обратные
    path_str = path_str.replace('/', '\\')

    # Ключевой момент: UNC-пути должны начинаться с \\
    # Если путь начинается с \ но не с \\ — добавляем ещё один
    if path_str.startswith('\\') and not path_str.startswith('\\\\'):
        path_str = '\\' + path_str

    # Убираем множественные слеши (больше двух) — оставляем максимум два в начале
    # Но сохраняем \\ в начале для UNC
    is_unc = path_str.startswith('\\\\')
    # Временно убираем начальные \\
    if is_unc:
        path_str = path_str[2:]

    # Заменяем множественные слеши на одинарные во всём пути
    while '\\\\' in path_str:
        path_str = path_str.replace('\\\\', '\\')

    # Возвращаем UNC-префикс
    if is_unc:
        path_str = '\\\\' + path_str

    # Проверяем, что путь похож на путь к файлу
    if len(path_str) < 5 or '.' not in path_str:
        return None

    return path_str

# ==================== ЧТЕНИЕ EXCEL ====================
print(f"📂 Читаю {INPUT_FILE}...")

# Читаем Excel, пропуская первые 4 строки, 5-я = заголовки
df = pd.read_excel(INPUT_FILE, header=4)

print(f"✅ Загружено строк: {len(df)}")
print(f"📋 Колонки: {df.columns.tolist()}")

# Поиск колонки с путями
path_col = None

# Способ 1: ищем по содержимому (UNC-пути или обычные)
for col in df.columns:
    sample = df[col].dropna().head(20).astype(str)
    if len(sample) > 0:
        # Ищем строки, похожие на пути (содержат \ и расширение файла)
        looks_like_path = sample.str.contains(r'\\') & sample.str.contains(r'\.(jpg|jpeg|png|bmp|gif)', case=False)
        if looks_like_path.any():
            path_col = col
            print(f"🎯 Нашёл пути в колонке: '{col}'")
            break

# Способ 2: последняя колонка
if path_col is None:
    path_col = df.columns[-1]
    print(f"🎯 Беру последнюю колонку: '{path_col}'")

# Способ 3: принудительно (если знаете номер)
if path_col is None:
    # Попробуйте раскомментировать и указать номер:
    # path_col = df.columns[7]  # 8-я колонка
    pass

if path_col is None:
    print("\n🔍 Содержимое ВСЕХ колонок (первые 3 строки):")
    for col in df.columns:
        vals = df[col].head(3).tolist()
        print(f"   '{col}': {vals}")
    raise ValueError("❌ Не могу найти колонку с путями!")

# ==================== ОБРАБОТКА ПУТЕЙ ====================
print(f"\n📂 Обрабатываю колонку: '{path_col}'")

# Применяем нормализацию
df['path_clean'] = df[path_col].apply(normalize_path)

# Убираем пустые
df = df[df['path_clean'].notna()].copy()

print(f"📸 Строк с заполненным путём: {len(df)}")

# Показываем примеры
print("📋 Примеры путей (первые 5):")
for i, p in enumerate(df['path_clean'].head(5)):
    print(f"   [{i+1}] {p}")
    print(f"       exists: {os.path.exists(p)}")

# ==================== ПРОВЕРКА ФАЙЛОВ ====================
print("\n🔍 Проверяю доступность файлов...")
existing = []
missing = []

for idx, row in df.iterrows():
    path = row['path_clean']
    if os.path.exists(path):
        existing.append(idx)
    else:
        missing.append(idx)

print(f"   Доступно: {len(existing)}")
print(f"   Не найдено: {len(missing)}")

if missing and len(missing) > 0:
    print(f"   Первые 5 недоступных:")
    for p in df.loc[missing[:5], 'path_clean']:
        print(f"      {p}")

df = df.loc[existing].copy()
total = len(df)
print(f"\n📸 Всего фото для проверки: {total}")

if total == 0:
    print("\n❌ Ни одно фото не доступно!")
    print("Возможные причины:")
    print("1. Сетевой диск не подключен (нужен доступ к \\\\Srv21\\Photo)")
    print("2. Права доступа")
    print("3. Пути повреждены в Excel")
    print("\nПопробуйте открыть этот путь в проводнике:")
    if len(missing) > 0:
        print(f"   {df.loc[missing[0], 'path_clean']}")
    exit(1)

# ==================== ЗАГРУЗКА МОДЕЛИ ====================
print(f"\n🧠 Загружаю модель: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("✅ Модель готова")

# ==================== ПАКЕТНАЯ ОБРАБОТКА ====================
print(f"\n🚀 Начинаю обработку (batch={BATCH_SIZE}, conf={CONF})...")

paths = df['path_clean'].tolist()
results_list = []

for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Батчи", unit="batch"):
    batch_paths = paths[i:i+BATCH_SIZE]

    try:
        batch_results = model.predict(
            source=batch_paths,
            conf=CONF,
            imgsz=IMG_SIZE,
            verbose=False,
            stream=False
        )

        for path, result in zip(batch_paths, batch_results):
            has_trash = False
            trash_info = []

            if result.masks is not None and len(result.boxes) > 0:
                has_trash = True
                for box in result.boxes:
                    trash_info.append({
                        'class': model.names[int(box.cls)],
                        'confidence': round(float(box.conf), 3)
                    })

            results_list.append({
                'path': path,
                'has_trash': has_trash,
                'detections': trash_info
            })

    except Exception as e:
        print(f"\n⚠️ Ошибка в батче {i//BATCH_SIZE}: {e}")
        for path in batch_paths:
            results_list.append({
                'path': path,
                'has_trash': None,
                'error': str(e)
            })

print(f"\n✅ Обработка завершена!")

# ==================== ФОРМИРОВАНИЕ РЕЗУЛЬТАТА ====================
print("📊 Формирую отчёт...")

results_df = pd.DataFrame(results_list)

# Объединяем с исходными данными
df_result = df.merge(results_df, left_on='path_clean', right_on='path', how='left')

# Оставляем только строки с найденным завалом
trash_df = df_result[df_result['has_trash'] == True].copy()

if len(trash_df) > 0:
    trash_df['classes_found'] = trash_df['detections'].apply(
        lambda d: ', '.join(set([x['class'] for x in d])) if d else ''
    )
    trash_df['max_confidence'] = trash_df['detections'].apply(
        lambda d: max([x['confidence'] for x in d]) if d else 0
    )
    trash_df['objects_count'] = trash_df['detections'].apply(
        lambda d: len(d) if d else 0
    )

    output_cols = []
    for col in df.columns:
        if col != 'path_clean':
            output_cols.append(col)

    output_cols.extend(['path_clean', 'has_trash', 'classes_found', 'max_confidence', 'objects_count'])
    output_cols = [c for c in output_cols if c in trash_df.columns]

    trash_out = trash_df[output_cols].copy()
    trash_out.to_excel(OUTPUT_FILE, index=False)

    print(f"\n{'='*50}")
    print(f"📊 СТАТИСТИКА:")
    print(f"   Всего проверено: {total}")
    print(f"   Найдено завалов: {len(trash_df)} ({len(trash_df)/total*100:.1f}%)")
    print(f"   Средняя уверенность: {trash_df['max_confidence'].mean():.3f}")
    print(f"   Всего объектов: {trash_df['objects_count'].sum()}")
    print(f"   Результат сохранён: {OUTPUT_FILE}")
    print(f"{'='*50}")

    print(f"\n🔝 ТОП-10 по уверенности:")
    top10 = trash_df.nlargest(10, 'max_confidence')[['path_clean', 'max_confidence', 'classes_found']]
    for _, row in top10.iterrows():
        fname = Path(row['path_clean']).name
        print(f"   {row['max_confidence']:.3f} | {row['classes_found']} | {fname}")
else:
    print(f"\n{'='*50}")
    print(f"✅ Завалов не найдено! Проверено {total} фото.")
    print(f"{'='*50}")
    pd.DataFrame({'Результат': ['Завалы не обнаружены']}).to_excel(OUTPUT_FILE, index=False)

print(f"\n🏁 Готово! Файл: {OUTPUT_FILE}")