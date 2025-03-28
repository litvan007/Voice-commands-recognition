import pyaudio
import wave
import os
import threading
import numpy as np
from scipy.signal import resample_poly

# Путь к выходному файлу (вы можете изменить на нужное)
WAVE_OUTPUT_FILENAME = os.path.expanduser("./output_new.wav")

# Параметры аудио (формат и количество каналов остаются прежними)
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024  # Размер буфера

# Инициализация PyAudio
audio = pyaudio.PyAudio()

# Вывод списка доступных аудиоустройств с информацией о частоте дискретизации
print("Доступные аудиоустройства:")
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    print(f"Индекс {i}: {info['name']} (макс. входных каналов: {info['maxInputChannels']}), "
          f"defaultSampleRate: {info['defaultSampleRate']}")

# Укажите нужный индекс устройства для записи (например, 2)
device_index = 2

# Получаем информацию о выбранном устройстве
device_info = audio.get_device_info_by_index(device_index)
# Запишем частоту, которую устройство поддерживает (нормально она передается как float)
original_rate = int(device_info['defaultSampleRate'])
print(f"Выбрано устройство {device_index}: {device_info['name']}, sample rate: {original_rate}")

# Открываем поток для записи с нативной частотой устройства
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=original_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

frames = []
stop_recording = False  # Флаг для остановки записи

def wait_for_key():
    """Ожидание нажатия Enter для остановки записи."""
    global stop_recording
    input("Нажмите Enter, чтобы остановить запись...\n")
    stop_recording = True

# Запускаем отдельный поток, который ждёт нажатия клавиши
key_thread = threading.Thread(target=wait_for_key)
key_thread.start()

print("Запись началась...")
while not stop_recording:
    try:
        # Чтение данных с микрофона с отключением исключения переполнения буфера
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    except Exception as e:
        print("Ошибка при чтении:", e)
        break

print("Запись завершена.")

# Останавливаем и закрываем поток, завершаем работу с PyAudio
stream.stop_stream()
stream.close()
audio.terminate()

# Объединяем все фреймы в один байтовый массив и преобразуем в numpy-массив int16
audio_data = b''.join(frames)
audio_np = np.frombuffer(audio_data, dtype=np.int16)

# Определяем целевую частоту дискретизации
target_rate = 16000

# Ресемплируем аудио с использованием функции resample_poly
# Параметры: up = target_rate, down = original_rate
resampled_audio = resample_poly(audio_np, target_rate, original_rate)

# Приводим результат к формату int16 с обрезкой значений до допустимого диапазона
resampled_audio_int16 = np.clip(resampled_audio, -32768, 32767).astype(np.int16)

# Сохраняем ресемплированные данные в WAV-файл с частотой 16000 Гц
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(target_rate)
    wf.writeframes(resampled_audio_int16.tobytes())

print(f"Файл сохранён: {WAVE_OUTPUT_FILENAME}")