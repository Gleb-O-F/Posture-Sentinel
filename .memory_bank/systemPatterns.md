# System Patterns: Posture Sentinel

## Архитектура
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Camera    │────▶│  MediaPipe  │────▶│   Posture   │
│  (opencv)   │     │   BlazePose │     │   Analysis  │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │   Overlay   │
                                       │  (opencv)   │
                                       └─────────────┘
```

## Паттерны
- **Pipeline**: Camera → MediaPipe → Posture → Overlay
- **Config**: YAML файл (config.yaml)
- **Model**: BlazePose 33 landmarks

## Поток данных
1. OpenCV захватывает frame
2. MediaPipe.process() → 33 landmarks
3. PostureAnalyzer: сравнение с baseline → статус
4. OpenCV: отрисовка скелета и текста

## Управление
- **c** - калибровка (сохранить текущую позу как baseline)
- **q** - выход
