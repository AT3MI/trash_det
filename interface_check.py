"""
Пакетная проверка фото на наличие завалов
Читает Excel, прогоняет через модель, сохраняет результат
с GUI-интерфейсом
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import traceback
import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading

# ==================== НАСТРОЙКИ (ЗАШИТЫ В КОД) ====================
# Путь к модели относительно папки с программой
MODEL_RELATIVE_PATH = os.path.join('runs', 'segment', 'train', 'weights', 'best.pt')
CONF = 0.3
IMG_SIZE = 640
BATCH_SIZE = 16

# ==================== НАСТРОЙКИ (ЗАШИТЫ В КОД) ====================
# Путь к модели относительно папки с программой
MODEL_RELATIVE_PATH = os.path.join('model_weights', 'best.pt')
CONF = 0.3
IMG_SIZE = 640
BATCH_SIZE = 16

# ==================== ОПРЕДЕЛЕНИЕ ПУТИ К МОДЕЛИ ====================
def get_model_path():
    """Определяет путь к модели (рядом с exe или скриптом)"""
    if getattr(sys, 'frozen', False):
        # Если запущено как exe (PyInstaller)
        base_dir = os.path.dirname(sys.executable)
    else:
        # Если запущено как скрипт
        base_dir = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_dir, MODEL_RELATIVE_PATH)

MODEL_PATH = get_model_path()

# Проверка: если модели нет — падаем сразу с понятным сообщением
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Модель не найдена: {MODEL_PATH}\n"
        f"Положите файл 'best.pt' в папку 'model_weights' рядом с программой.\n"
        f"Скачать можно здесь: https://huggingface.co/AT3MI/trash-detector-yolo11s"
    )

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

# ==================== ГЛАВНЫЙ КЛАСС ПРИЛОЖЕНИЯ ====================
class TrashDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детектор завалов v1.0")
        self.root.geometry("750x550")
        self.root.resizable(True, True)

        # Переменные
        self.input_file = tk.StringVar(value='выполнено с фото из путевого.xlsx')
        self.processing = False

        # Создаём интерфейс
        self.create_widgets()

        # Проверяем наличие модели при запуске
        self.check_model()

    def check_model(self):
        """Проверяет наличие файла модели"""
        if not os.path.exists(MODEL_PATH):
            self.log(f"⚠️ ВНИМАНИЕ: Модель не найдена!", "warning")
            self.log(f"   Ожидаемый путь: runs/segment/train/weights/best.pt", "warning")
            self.log(f"   Убедитесь, что папка runs лежит рядом с программой", "warning")
            self.log(f"   и содержит best.pt внутри\n", "warning")
        else:
            self.log(f"✅ Модель найдена", "success")

    def create_widgets(self):
        # Главный контейнер с отступами
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ==================== БЛОК ВЫБОРА ФАЙЛА ====================
        file_frame = ttk.LabelFrame(main_frame, text="Исходный файл", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(file_frame, text="Excel-файл с путями к фото:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Entry(file_frame, textvariable=self.input_file, width=55).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(file_frame, text="Обзор...", command=self.browse_input, width=10).grid(row=0, column=2)

        # ==================== ИНФОРМАЦИЯ О ПАРАМЕТРАХ ====================
        info_frame = ttk.LabelFrame(main_frame, text="Параметры обработки", padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        info_text = f"Порог уверенности: {CONF}\n"
        info_text += f"Размер изображения: {IMG_SIZE}px\n"
        info_text += f"Размер батча: {BATCH_SIZE}"

        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).grid(row=0, column=0, sticky=tk.W)

        # ==================== КНОПКИ УПРАВЛЕНИЯ ====================
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.btn_start = ttk.Button(buttons_frame, text="🚀 ЗАПУСТИТЬ ПРОВЕРКУ", command=self.start_processing, width=25)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_stop = ttk.Button(buttons_frame, text="⏹ ОСТАНОВИТЬ", command=self.stop_processing, state=tk.DISABLED, width=15)
        self.btn_stop.pack(side=tk.LEFT)

        # ==================== ПРОГРЕСС-БАР ====================
        progress_frame = ttk.LabelFrame(main_frame, text="Прогресс", padding="10")
        progress_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=500)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        self.progress_label = ttk.Label(progress_frame, text="Готов к работе")
        self.progress_label.pack()

        # ==================== ЛОГ ====================
        log_frame = ttk.LabelFrame(main_frame, text="Лог выполнения", padding="5")
        log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Настройка тегов для цветного текста
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("warning", foreground="orange")
        self.log_text.tag_config("info", foreground="blue")
        self.log_text.tag_config("bold", font=("TkDefaultFont", 9, "bold"))

        # ==================== СТАТУС-БАР ====================
        self.status_var = tk.StringVar(value="Готов")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, sticky=(tk.W, tk.E))

        # Настройка растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

    def log(self, message, tag=None):
        """Добавляет сообщение в лог"""
        self.log_text.config(state=tk.NORMAL)
        if tag:
            self.log_text.insert(tk.END, message + "\n", tag)
        else:
            self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Выберите Excel-файл",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)

    def start_processing(self):
        if self.processing:
            return

        # Проверяем входной файл
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Ошибка", f"Файл не найден:\n{self.input_file.get()}")
            return

        # Проверяем модель
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Ошибка",
                f"Модель не найдена!\n\n"
                f"Убедитесь, что рядом с программой\n"
                f"есть папка runs, а внутри неё:\n"
                f"runs/segment/train/weights/best.pt")
            return

        self.processing = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.config(text="Инициализация...")

        # Запускаем обработку в отдельном потоке
        thread = threading.Thread(target=self.process, daemon=True)
        thread.start()

    def stop_processing(self):
        self.processing = False
        self.log("⚠️ Отправлен сигнал остановки...", "warning")
        self.btn_stop.config(state=tk.DISABLED)

    def update_progress(self, value, text):
        """Обновляет прогресс-бар и текст"""
        self.progress_var.set(value)
        self.progress_label.config(text=text)
        self.status_var.set(text)
        self.root.update_idletasks()

    def process(self):
        try:
            # ==================== ЧТЕНИЕ EXCEL ====================
            self.log(f"📂 Читаю {self.input_file.get()}...", "info")
            self.update_progress(5, "Чтение Excel...")

            df = pd.read_excel(self.input_file.get(), header=4)
            self.log(f"✅ Загружено строк: {len(df)}", "success")
            self.log(f"📋 Колонки: {df.columns.tolist()}", "info")

            # Поиск колонки с путями
            path_col = None
            for col in df.columns:
                sample = df[col].dropna().head(20).astype(str)
                if len(sample) > 0:
                    looks_like_path = sample.str.contains(r'\\') & sample.str.contains(r'\.(jpg|jpeg|png|bmp|gif)', case=False)
                    if looks_like_path.any():
                        path_col = col
                        self.log(f"🎯 Нашёл пути в колонке: '{col}'", "success")
                        break

            if path_col is None:
                path_col = df.columns[-1]
                self.log(f"🎯 Беру последнюю колонку: '{path_col}'", "warning")

            if path_col is None:
                self.log("❌ Не могу найти колонку с путями!", "error")
                messagebox.showerror("Ошибка", "Не могу найти колонку с путями к фото!")
                return

            # Нормализация путей
            self.update_progress(10, "Обработка путей...")
            df['path_clean'] = df[path_col].apply(normalize_path)
            df = df[df['path_clean'].notna()].copy()
            self.log(f"📸 Строк с заполненным путём: {len(df)}", "info")

            # Проверка доступности
            self.update_progress(15, "Проверка доступности файлов...")
            existing = []
            missing = []

            for idx, row in df.iterrows():
                if not self.processing:
                    self.log("⏹ Обработка остановлена пользователем", "warning")
                    return

                if os.path.exists(row['path_clean']):
                    existing.append(idx)
                else:
                    missing.append(idx)

            self.log(f"   Доступно: {len(existing)}", "success")
            self.log(f"   Не найдено: {len(missing)}", "error" if missing else "success")

            df = df.loc[existing].copy()
            total = len(df)
            self.log(f"\n📸 Всего фото для проверки: {total}", "info")

            if total == 0:
                self.log("❌ Ни одно фото не доступно!", "error")
                messagebox.showerror("Ошибка", "Ни одно фото не доступно!\nПроверьте сетевые диски и права доступа.")
                return

            # Загрузка модели
            self.update_progress(20, "Загрузка модели...")
            self.log(f"\n🧠 Загружаю модель...", "info")
            model = YOLO(MODEL_PATH)
            self.log("✅ Модель готова", "success")

            # Обработка
            self.log(f"\n🚀 Начинаю обработку (batch={BATCH_SIZE}, conf={CONF})...", "info")

            paths = df['path_clean'].tolist()
            results_list = []
            total_batches = (len(paths) + BATCH_SIZE - 1) // BATCH_SIZE

            for i in range(0, len(paths), BATCH_SIZE):
                if not self.processing:
                    self.log("⏹ Обработка остановлена пользователем", "warning")
                    return

                batch_paths = paths[i:i+BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1

                # Прогресс: 20-90%
                progress = 20 + (batch_num / total_batches) * 70
                self.update_progress(progress, f"Обработка батча {batch_num}/{total_batches}...")
                self.log(f"⚙️ Батч {batch_num}/{total_batches} ({len(batch_paths)} фото)...", "info")

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
                    self.log(f"⚠️ Ошибка в батче {batch_num}: {e}", "error")
                    for path in batch_paths:
                        results_list.append({
                            'path': path,
                            'has_trash': None,
                            'error': str(e)
                        })

            if not self.processing:
                return

            # Формирование результата
            self.update_progress(95, "Формирование отчёта...")
            self.log("\n📊 Формирую отчёт...", "info")

            results_df = pd.DataFrame(results_list)
            df_result = df.merge(results_df, left_on='path_clean', right_on='path', how='left')
            trash_df = df_result[df_result['has_trash'] == True].copy()

            # Определяем папку для сохранения (рядом с программой)
            if getattr(sys, 'frozen', False):
                save_dir = os.path.dirname(sys.executable)
            else:
                save_dir = os.path.dirname(os.path.abspath(__file__))

            output_file = os.path.join(save_dir, f'площадки_с_завалами_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')

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
                trash_out.to_excel(output_file, index=False)

                self.log(f"\n{'='*50}", "info")
                self.log(f"📊 СТАТИСТИКА:", "bold")
                self.log(f"   Всего проверено: {total}", "info")
                self.log(f"   Найдено завалов: {len(trash_df)} ({len(trash_df)/total*100:.1f}%)", "success")
                self.log(f"   Средняя уверенность: {trash_df['max_confidence'].mean():.3f}", "info")
                self.log(f"   Всего объектов: {trash_df['objects_count'].sum()}", "info")
                self.log(f"   Результат сохранён: {os.path.basename(output_file)}", "success")
            else:
                self.log(f"\n{'='*50}", "info")
                self.log(f"✅ Завалов не найдено! Проверено {total} фото.", "success")
                pd.DataFrame({'Результат': ['Завалы не обнаружены']}).to_excel(output_file, index=False)

            self.update_progress(100, "Готово!")
            self.log(f"\n🏁 Готово! Файл: {os.path.basename(output_file)}", "success")

            # Открываем папку с результатом
            if messagebox.askyesno("Готово", f"Обработка завершена!\n\nФайл сохранён:\n{os.path.basename(output_file)}\n\nОткрыть папку с файлом?"):
                os.startfile(save_dir)

        except Exception as e:
            self.log(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}", "error")
            self.log(traceback.format_exc(), "error")
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")

        finally:
            self.processing = False
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.status_var.set("Готов")

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = TrashDetectorApp(root)
    root.mainloop()