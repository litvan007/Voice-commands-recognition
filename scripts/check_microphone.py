import pyaudio
import numpy as np
import time
import sys

def check_microphone(device_index=2):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    
    try:
        # Проверяем устройство
        device_info = p.get_device_info_by_index(device_index)
        print(f"\nПроверка микрофона:")
        print(f"Устройство: {device_info['name']}")
        print(f"Частота дискретизации: {RATE} Hz")
        print(f"Входных каналов: {device_info['maxInputChannels']}")
        
        if device_info['maxInputChannels'] == 0:
            print("Ошибка: Устройство не поддерживает ввод!")
            return
            
        # Открываем поток
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       input_device_index=device_index,
                       frames_per_buffer=CHUNK)
        
        print("\nНачинаю слушать микрофон...")
        print("Нажмите Ctrl+C для выхода")
        print("Уровень сигнала (чем громче звук, тем больше значение):")
        
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data**2))
                
                # Создаем простую визуализацию уровня
                level = int(rms / 100)  # Масштабируем для отображения
                bar = '█' * min(level, 50)  # Ограничиваем длину полоски
                print(f"\rУровень: {rms:8.2f} |{bar:<50}|", end='')
                
            except KeyboardInterrupt:
                print("\nЗавершение...")
                break
                
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        device_index = int(sys.argv[1])
        check_microphone(device_index)
    else:
        check_microphone() 