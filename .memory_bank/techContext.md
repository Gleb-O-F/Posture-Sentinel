# Tech Context: Posture Sentinel

## Стек технологий
| Компонент | Технология |
|-----------|------------|
| Язык | Rust (stable) |
| CV | opencv 0.92 |
| ML | ort 2.0.0-rc.11 (ONNX Runtime) |
| UI | winit 0.29 |
| Config | serde + toml |

## Зависимости
- opencv (с clang-runtime)
- ort с CUDA features
- winit
- anyhow, thiserror

## Сборка
Требует:
- Visual Studio Build Tools + C++ компоненты
- CUDA Toolkit
- clang

## Модели
- `models/pose_estimation_mediapipe.onnx`
- `models/pose_landmarks_detector_full.onnx`
- `models/pose_detection.onnx`

## Конфигурация
- Файл: `config.toml`
- Параметры: камера, пороги детекции, настройки GPU
