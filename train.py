from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Загружаем базовую модель для сегментации (она сама скачается при первом запуске)
    model = YOLO('yolo11s-seg.pt')

    # 2. Запускаем обучение
    model.train(
        data='data.yaml',
        epochs=50,        # Начни с 50. Если будет мало, потом увеличишь до 100
        imgsz=640,
        batch=8,          # Если будет ошибка "Out of memory", уменьши до 4 или 2
        task='segment',   # Обязательно, так как у нас полигоны
        device='cpu',     # Если есть видеокарта NVIDIA, поменяй на '0'
        workers=0,        # ВАЖНО для Windows! Если будет ошибка DataLoader, ставь 0
        plots=True        # Будет рисовать графики обучения
    )