import pyaudio
import wave
import numpy as np
import time
from voice.audio_input import list_audio_devices

def test_microphone(device_index=2, duration=5):
    # Сначала покажем список доступных устройств
    list_audio_devices()
    
    # Параметры записи
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    
    try:
        # Проверяем устройство
        device_info = p.get_device_info_by_index(device_index)
        if device_info['maxInputChannels'] == 0:
            print(f"Ошибка: Устройство {device_index} не поддерживает ввод!")
            return
            
        print(f"\nТестирование записи с устройства {device_index}...")
        print(f"Имя устройства: {device_info['name']}")
        print(f"Частота дискретизации: {RATE} Hz")
        
        # Открываем поток
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       input_device_index=device_index,
                       frames_per_buffer=CHUNK)
        
        print(f"\nЗапись в течение {duration} секунд...")
        frames = []
        
        # Записываем
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Выводим уровень сигнала
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data**2))
            print(f"Уровень сигнала: {rms:.2f}", end='\r')
        
        print("\nЗапись завершена!")
        
        # Сохраняем в файл
        wf = wave.open("test_recording.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print("Файл сохранен как test_recording.wav")
        
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    test_microphone() 