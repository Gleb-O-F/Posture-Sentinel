# Tech Context: Posture Sentinel

## Стек технологий
| Компонент | Технология |
|-----------|------------|
| Язык | Python 3.11+ |
| CV | opencv-python |
| ML | MediaPipe |
| Модель | BlazePose |

## Зависимости
```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

## Установка
```bash
pip install -r requirements.txt
```

## Запуск
```bash
python main.py
```

## Управление
- **c** - калибровка (встаньте в правильную позу)
- **q** - выход

## Конфигурация
- Файл: `config.yaml` (создается после калибровки)
