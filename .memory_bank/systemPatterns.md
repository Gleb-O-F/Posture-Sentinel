# System Patterns: Posture Sentinel

## Архитектура
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Camera    │────▶│  Inference  │────▶│   Posture   │
│  (opencv)   │     │    (ort)    │     │   Analysis  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Overlay   │
                                        │  (winit)    │
                                        └─────────────┘
```

## Паттерны
- **Pipeline**: Camera → Inference → Posture → Overlay
- **Config**: TOML файл (config.toml)
- **Model**: BlazePose 33 landmarks (MediaPipe)

## Границы модулей
| Модуль | Ответственность |
|--------|----------------|
| `camera` | Захват кадров с веб-камеры, управление потоком |
| `inference` | Загрузка ONNX, препроцессинг, инференс |
| `posture` | Детекция нарушений осанки, калибровка |
| `config` | Чтение/запись config.toml |
| `overlay` | Отрисовка скелета, размытие экрана |

## Поток данных
1. Камера захватывает frame (opencv::Mat)
2. Inference: препроцессинг → ort::Session → получение landmarks
3. Posture: сравнение с baseline → статус осанки
4. Overlay: отрисовка результатов в окно
