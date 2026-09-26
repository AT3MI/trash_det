from ultralytics import YOLO

# 1. Загружаем твою обученную модель
model = YOLO('runs/segment/train/weights/best.pt')

# 2. Запускаем предсказание
results = model.predict(
    source='IMG_20260624_102639.png',  # <-- ЗАМЕНИ НА ИМЯ ТВОЕГО ФОТО
    save=True,                # Сохранить картинку с нарисованными масками
    conf=0.3,                 # Порог уверенности (0.5 = 50%)
    imgsz=640
)

print("Готово! Картинка с масками сохранена в папке: runs/segment/predict/")