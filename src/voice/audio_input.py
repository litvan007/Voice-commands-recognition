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
logger.setLevel(logging.DEBUG)  # можно сделать DEBUG при отладке
logging.getLogger("pyaudio").setLevel(logging.WARNING)


def prepare_audio_device(device_index: int):
    """
    Один раз инициализирует PyAudio и возвращает выбранное устройство.
    """
    audio = pyaudio.PyAudio()
    device_info = audio.get_device_info_by_index(device_index)

    print("🎤 Используется аудиоустройство:")
    print(f"[{device_index}] {device_info['name']} @ {device_info['defaultSampleRate']} Hz")

    return audio, device_index, device_info


class AudioRecorder:
    def __init__(self, audio, device_index, device_info, sample_rate=16000, desired_langth=96000, channels=1, chunk=1024):
        self.audio = audio
        self.device_index = device_index
        self.device_info = device_info
        self.original_rate = int(device_info['defaultSampleRate'])
        self.target_rate = sample_rate
        self.desired_langth = desired_langth
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
        audio_data = b''.join(self.frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        if self.original_rate != self.target_rate:
            logger.debug(f"Ресемплирование: {self.original_rate} → {self.target_rate}")
            resampled = resample_poly(audio_np, self.target_rate, self.original_rate)
            audio_np = np.clip(resampled, -32768, 32767).astype(np.int16)

        audio_np = audio_np.astype(np.float32) / 32768.0

        valid_frames = audio_np.shape[0]
        if valid_frames < self.desired_length:
            pad_length = self.desired_length - valid_frames

            # Берем первые 5 отсчетов сигнала
            first_five = audio_np[:100]
            # Заполняем недостающую часть, выбирая случайным образом из первых 5 отсчетов
            pad_segment = np.random.choice(first_five, size=pad_length, replace=True)
            padded_signal = np.concatenate([audio_np, pad_segment])

        return padded_signal, valid_frames
    
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
