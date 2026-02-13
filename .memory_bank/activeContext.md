# Active Context: Posture Sentinel

## Текущий фокус
**Python MVP** — базовая версия с детекцией осанки.

## Активные решения
- Язык: Python 3.11+
- CV: opencv-python
- ML: MediaPipe (BlazePose)
- Отрисовка: OpenCV

## Структура проекта
- `main.py` — основное приложение
- `requirements.txt` — зависимости
- `config.yaml` — конфиг (создается при калибровке)

## Приоритеты
1. Тестирование текущей версии
2. Улучшение детекции (slouching, forward lean, head tilt)
3. Добавить эффект размытия
4. Сохранение калибровки

## Известные неопределенности
- Как реализовать размытие экрана? (MediaPipe не имеет built-in)
