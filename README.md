# Posture Sentinel

AI-приложение для мониторинга осанки в реальном времени.

## Возможности

- Детекция 33 ключевых точек тела (BlazePose)
- Мониторинг осанки (сутулость, наклон плечей)
- GPU ускорение через DirectML (Windows)
- Визуализация скелета
- Сглаживание для стабильности
- Калибровка базового положения

## Требования

- Python 3.11+
- OpenCV: `pip install opencv-python`
- ONNX Runtime: `pip install onnxruntime-directml`

## Установка

```bash
pip install opencv-python onnxruntime-directml
```

## Запуск

```bash
python run.py
```

## Управление

- **c** - калибровка (встаньте в правильную позу и нажмите)
- **q** - выход

## Модели

Модели ONNX находятся в папке `models/`:
- `pose_estimation_mediapipe.onnx` - основная модель

## Производительность

- FPS: ~20-30 на GPU (DirectML)
- Quadro P1000: ~15-20 FPS

## Rust версия (в разработке)

Требует Visual Studio Build Tools + C++ компоненты + CUDA. 
