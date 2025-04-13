import librosa
import numpy as np
import logging
from scipy.fftpack import dct
from scipy.signal import resample_poly

import os
import datetime
import pyaudio
import wave
import threading
import logging

import opensmile

import time

from settings import Settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Уровень будет контролироваться в main.py

# Отключаем логирование для других модулей
logging.getLogger("pyaudio").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("librosa").setLevel(logging.WARNING)
logging.getLogger("sounddevice").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def list_audio_devices():
    """
    Выводит список всех доступных аудиоустройств с их параметрами.
    """
    audio = pyaudio.PyAudio()
    print("\nДоступные аудиоустройства:")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        print(f"\nУстройство {i}:")
        print(f"  Имя: {info['name']}")
        print(f"  Макс. входных каналов: {info['maxInputChannels']}")
        print(f"  Частота дискретизации: {info['defaultSampleRate']} Hz")
        print(f"  Поддерживает ввод: {'Да' if info['maxInputChannels'] > 0 else 'Нет'}")
    audio.terminate()


def prepare_audio_device(device_index: int):
    """
    Один раз инициализирует PyAudio и возвращает выбранное устройство.
    """
    audio = pyaudio.PyAudio()
    device_info = audio.get_device_info_by_index(device_index)

    if device_info['maxInputChannels'] == 0:
        raise ValueError(f"Устройство {device_index} не поддерживает ввод аудио!")

    print("\n🎤 Используется аудиоустройство:")
    print(f"[{device_index}] {device_info['name']}")
    print(f"Частота дискретизации: {device_info['defaultSampleRate']} Hz")
    print(f"Входных каналов: {device_info['maxInputChannels']}")
    print(f"Формат: {device_info['defaultSampleFormat']}")
    print(f"Размер буфера: {device_info['defaultLowInputLatency']}")

    return audio, device_index, device_info


class AudioRecorder:
    def __init__(self, audio, device_index, device_info, sample_rate=16000, desired_length=96000, channels=1, chunk=1024):
        self.audio = audio
        self.device_index = device_index
        self.device_info = device_info
        self.original_rate = int(device_info['defaultSampleRate'])
        self.target_rate = sample_rate
        self.desired_length = desired_length
        self.channels = channels
        self.chunk = chunk
        self.format = pyaudio.paInt16
        self.frames = []
        self.stop_recording = False

    def start_recording(self):
        logger.info(f"🎙 Запись с устройства [{self.device_index}]: {self.device_info['name']} @ {self.original_rate} Гц")
        stream = self.audio.open(format=self.format,
                                 channels=self.channels,
                                 rate=self.original_rate,
                                 input=True,
                                 input_device_index=self.device_index,
                                 frames_per_buffer=self.chunk)

        def wait_for_key():
            input("Нажмите Enter для остановки...\n")
            self.stop_recording = True

        threading.Thread(target=wait_for_key).start()

        while not self.stop_recording:
            data = stream.read(self.chunk, exception_on_overflow=False)
            self.frames.append(data)

        stream.stop_stream()
        stream.close()
        logger.info("📥 Запись завершена")

    def get_resampled_audio(self) -> np.ndarray:
        if not self.frames:
            logger.error("Нет аудиоданных для обработки")
            return np.zeros(self.desired_length), 0

        audio_data = b''.join(self.frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        logger.debug(f"Максимальное значение до ресемплирования: {np.max(np.abs(audio_np))}")

        if self.original_rate != self.target_rate:
            logger.debug(f"Ресемплирование: {self.original_rate} → {self.target_rate}")
            resampled = resample_poly(audio_np, self.target_rate, self.original_rate)
            
            # Нормализуем значения перед клиппингом
            max_val = np.max(np.abs(resampled))
            if max_val > 0:
                resampled = resampled / max_val
            
            audio_np = np.clip(resampled, -1.0, 1.0)
            logger.debug(f"Максимальное значение после ресемплирования: {np.max(np.abs(audio_np))}")

        valid_frames = audio_np.shape[0]
        if valid_frames < self.desired_length:
            pad_length = self.desired_length - valid_frames
            logger.debug(f"Добавление паддинга: {pad_length} сэмплов")
            
            if valid_frames > 100:
                last_hundred = audio_np[-100:]
                pad_segment = np.random.choice(last_hundred, size=pad_length, replace=True)
            else:
                pad_segment = np.zeros(pad_length)
            
            padded_signal = np.concatenate([audio_np, pad_segment])
            return padded_signal, valid_frames
        else:
            return audio_np, valid_frames
    
    def save_to_wav(self, signal: np.ndarray, directory: str = "debug_wavs"):
        """
        Сохраняет текущую запись в WAV-файл с уникальным именем.
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recognized_{timestamp}.wav"
        path = os.path.join(directory, filename)

        audio_int16 = (np.clip(
            np.array(signal) * 32768, -32768, 32767
        )).astype(np.int16)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.target_rate)
            wf.writeframes(audio_int16.tobytes())

        logger.debug(f"🎙️ Аудио сохранено: {path}")

class RingBufferAudioRecorder:
    def __init__(self, audio, device_index, device_info, sample_rate=16000, buffer_duration=6, update_interval=1, channels=1, chunk=1024):
        self.audio = audio
        self.device_index = device_index
        self.device_info = device_info
        self.original_rate = int(device_info['defaultSampleRate'])
        self.target_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.format = pyaudio.paInt16
        
        # Параметры кольцевого буфера
        self.buffer_duration = buffer_duration  # длительность буфера в секундах
        self.update_interval = update_interval  # интервал обновления в секундах
        
        # Размер буфера в сэмплах
        self.buffer_size = int(self.buffer_duration * self.original_rate)
        self.update_size = int(self.update_interval * self.original_rate)
        
        # Инициализация буфера
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.current_position = 0
        self.is_recording = False
        self.stream = None
        self.recording_thread = None
        self.last_vad_check = 0

    def start_recording(self):
        logger.info(f"🎙 Запись с устройства [{self.device_index}]: {self.device_info['name']} @ {self.original_rate} Гц")
        
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.original_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk,
            stream_callback=self._callback
        )
        
        self.is_recording = True
        self.stream.start_stream()
        self.last_vad_check = time.time()

    def stop_recording(self):
        if self.stream is not None:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            logger.info("📥 Запись завершена")

    def _callback(self, in_data, frame_count, time_info, status):
        if not self.is_recording:
            return (None, pyaudio.paComplete)
        
        # Преобразование входных данных в numpy массив
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Обновление буфера - добавляем новые данные в конец
        if self.current_position + len(audio_data) <= self.buffer_size:
            self.buffer[self.current_position:self.current_position + len(audio_data)] = audio_data
        else:
            # Если данные не помещаются в конец буфера, переносим их в начало
            remaining = self.buffer_size - self.current_position
            self.buffer[self.current_position:] = audio_data[:remaining]
            self.buffer[:len(audio_data) - remaining] = audio_data[remaining:]
        
        self.current_position = (self.current_position + len(audio_data)) % self.buffer_size
        
        return (None, pyaudio.paContinue)

    def get_current_buffer(self):
        """Возвращает текущее содержимое буфера в правильном порядке"""
        if self.current_position == 0:
            return self.buffer.copy()
        
        # Переупорядочиваем буфер так, чтобы последние данные были в конце
        ordered_buffer = np.zeros_like(self.buffer)
        ordered_buffer[:self.buffer_size - self.current_position] = self.buffer[self.current_position:]
        ordered_buffer[self.buffer_size - self.current_position:] = self.buffer[:self.current_position]
        return ordered_buffer

    def get_resampled_audio(self) -> np.ndarray:
        """Возвращает текущее содержимое буфера с ресемплированием и паддингом"""
        current_buffer = self.get_current_buffer()
        
        if self.original_rate != self.target_rate:
            logger.debug(f"Ресемплирование: {self.original_rate} → {self.target_rate}")
            resampled = resample_poly(current_buffer, self.target_rate, self.original_rate)
            current_buffer = np.clip(resampled, -1.0, 1.0)
        
        # Если сигнал короче желаемой длины, добавляем паддинг из начала
        if len(current_buffer) < self.buffer_size:
            pad_length = self.buffer_size - len(current_buffer)
            if len(current_buffer) > 100:  # Если есть достаточно данных для паддинга
                first_five = current_buffer[:100]
                pad_segment = np.random.choice(first_five, size=pad_length, replace=True)
                current_buffer = np.concatenate([current_buffer, pad_segment])
            else:
                # Если данных слишком мало, просто заполняем нулями
                current_buffer = np.pad(current_buffer, (0, pad_length), mode='constant')
        
        return current_buffer, len(current_buffer)

    def clear_buffer(self):
        """Очищает буфер"""
        self.buffer.fill(0)
        self.current_position = 0

    def should_check_vad(self) -> bool:
        """Проверяет, нужно ли запускать VAD"""
        current_time = time.time()
        if current_time - self.last_vad_check >= self.update_interval:
            self.last_vad_check = current_time
            return True
        return False

class FeaturesAudio:
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.emobase,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

    def sample(self, wave_path: str, settings: Settings): # OLD
        sample_rate = settings.audio_config.common.sample_rate
        logger.info(f"Загрузка аудио: {wave_path} с sample_rate={sample_rate}")

        signal, _ = librosa.load(wave_path, sr=sample_rate)

        if signal.dtype == np.int16:
            logger.debug("Конвертация int16 → float32")
            signal = signal.astype(np.float32) / 32768.0

        if signal.ndim > 1:
            logger.debug("Аудио многоканальное, берём первый канал")
            signal = signal[:, 0]

        logger.info(f"Аудио успешно загружено, длина = {len(signal)} сэмплов")
        return signal, sample_rate

    def get_features(self, signal: np.ndarray, settings: Settings, feature_type: str):
        if feature_type == 'VCR':
            cfg = settings.audio_config.command_features
            sample_rate = settings.audio_config.common.sample_rate

            logger.info("Извлечение VCR-признаков (MFCC)")

            mel_spec = librosa.feature.melspectrogram(
                y=signal,
                sr=sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.n_fft,
                window=cfg.window or 'hann',
                center=cfg.center,
                pad_mode='reflect',
                power=cfg.power,
                n_mels=cfg.n_mels,
                fmin=cfg.fmin,
                fmax=cfg.fmax,
                htk=cfg.htk,
                norm=None
            )


            mel_spec = np.maximum(mel_spec, 1e-10)
            mel_spec_db = 10.0 * np.log10(mel_spec / mel_spec.max())

            mfcc = dct(mel_spec_db, type=2, axis=0, norm='ortho')[:cfg.n_mfcc, :]
            logger.info(f"MFCC успешно извлечены: shape = {mfcc.shape}")
            return mfcc

        elif feature_type == 'VAD':
            logger.info("Извлечение VAD-фичей через openSMILE")

            sample_rate = settings.audio_config.common.sample_rate
            features_df = self.smile.process_signal(signal, 16000)
            features_np = features_df.to_numpy()

            logger.info(f"VAD-фичи извлечены: shape = {features_np.shape}")
            return features_np
