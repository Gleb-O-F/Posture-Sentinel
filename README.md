# Posture Sentinel

AI-приложение для мониторинга осанки в реальном времени.

## Возможности

- Детекция 33 ключевых точек тела (MediaPipe BlazePose)
- Мониторинг осанки (сутулость, наклон вперед, наклон головы)
- Постепенное размытие экрана при нарушениях
- Калибровка под индивидуальную посадку
- Сохранение настроек в `config.yaml`

## Требования

- Python 3.10+
- Камера (веб-камера ноутбука или USB)

## Установка

1. Клонируйте репозиторий
2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Запуск

```bash
python main.py
```

Фоновый режим без preview-окна:

```bash
python main.py --headless
```

Полностью без tray (для сервисного запуска из консоли):

```bash
python main.py --headless --no-tray
```

## Управление

- **c** - калибровка (примите правильную позу и нажмите)
- **q** - выход из приложения
- В tray-меню:
  - `Auto Start` - включить/выключить автозапуск Windows
  - `Auto Tune` - включить/выключить авто-подстройку порогов
  - `Auto-tune Thresholds` - запустить подстройку порогов вручную по логам
  - `Export Daily Summary` - выгрузить сводку за текущий день
  - автозапуск запускает приложение в `--headless` режиме

## Производительность и GPU

Текущая версия использует **ONNX Runtime** c приоритетом **DirectML (GPU)** и fallback на **CPU**.

Что реализовано:
- Автопереподключение камеры при потере сигнала.
- Мониторинг FPS по окну времени.
- Автоматический переход с DirectML на CPU при низком FPS.
- Индикация `Provider/FPS` в preview и provider в системном трее.

## Логи нарушений

При подтверждённом нарушении создаётся запись в:
- `logs/YYYY-MM-DD.jsonl`

Формат записи:
- `timestamp`
- `violation`
- `duration_sec`
- `camera_status`

## Дневная сводка

Сводка за день сохраняется в:
- `logs/YYYY-MM-DD.summary.json`

Содержимое сводки:
- общее число нарушений и суммарная длительность
- разбивка по типам нарушений (`count`, `total_duration_sec`, `avg_duration_sec`)
- статистика статусов камеры

Экспорт:
- через tray-меню: `Export Daily Summary`
- автоматически при выходе из приложения

## CLI отчёты

Можно строить сводки без запуска GUI:

```bash
python tools/report.py --date 2026-03-06
python tools/report.py --from 2026-03-01 --to 2026-03-06
python tools/report.py --date 2026-03-06 --print
```

## Auto-tuning порогов

Авто-подстройка использует логи последних дней и мягко изменяет:
- `slouch_threshold`
- `lean_threshold`
- `shoulder_tilt_threshold`
- `shoulder_width_threshold`

Ограничения:
- минимум 3 дня с логами
- минимум 15 событий
- изменение не более 20% за один запуск
- безопасные границы каждого порога

CLI-запуск:

```bash
python tools/tune.py --dry-run --print
python tools/tune.py
python tools/tune.py --lookback-days 7 --target-events 10
```
