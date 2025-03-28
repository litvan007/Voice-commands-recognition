# Voice-commands-recognition

Диплом, который состоит также из курсовых работ.

Управление механическим манипулятором голосом с помощью моделей для распознавания команд и детекции голосовой активности.

## ⚙️ Структура

- `models/` — предобученные модели в ONNX (`vad.onnx`, `command_recognizer.onnx`)
- `src/voice/` — обработка аудио, VAD, распознавание команд
- `src/control/` — преобразование команд в действия и управление манипулятором
- `configs/` — YAML-конфиги для меток и аудиопараметров
- `scripts/` — отладочные скрипты
- `main.py` — основной модуль: поток от аудио до движения

## 🚀 Быстрый старт

1. Установи зависимости:
```bash
pip install -r requirements.txt

* Ссылка на курсач: https://drive.google.com/file/d/11rhhHlpCGn5K0mW3c9JeV_xrkdxU3kJA/view?usp=sharing

* Ссылка на презентацию: https://drive.google.com/file/d/1tvHh7ueNbYsCHxUspxGmCxzYl4MTwVoK/view?usp=sharing

