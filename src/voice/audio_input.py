import librosa
import numpy as np
import logging
from scipy.fftpack import dct
from scipy.signal import resample_poly

import pyaudio
import wave
import threading
import logging

from settings import Settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # можно сделать DEBUG при отладке

class AudioRecorder:
    def __init__(self, device_index: int = 0, sample_rate: int = 16000, channels: int = 1, chunk: int = 1024):
        self.device_index = device_index
        self.target_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.format = pyaudio.paInt16
        self.frames = []
        self.stop_recording = False

        self.audio = pyaudio.PyAudio()
        self.device_info = self.audio.get_device_info_by_index(self.device_index)
        self.original_rate = int(self.device_info['defaultSampleRate'])

    def list_devices(self):
        print("Доступные аудиоустройства:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            print(f"[{i}] {info['name']} (входы: {info['maxInputChannels']}), {info['defaultSampleRate']} Гц")

    def start_recording(self):
        logger.info(f"Запись с устройства {self.device_index}: {self.device_info['name']} @ {self.original_rate} Гц")
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
        self.audio.terminate()
        logger.info("Запись завершена")

    def get_resampled_audio(self) -> np.ndarray:
        audio_data = b''.join(self.frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)

        if self.original_rate != self.target_rate:
            logger.debug(f"Ресемплирование: {self.original_rate} → {self.target_rate}")
            resampled = resample_poly(audio_np, self.target_rate, self.original_rate)
            audio_np = np.clip(resampled, -32768, 32767).astype(np.int16)

        # Приводим к float32, как делает librosa
        return audio_np.astype(np.float32) / 32768.0


    def save_to_wav(self, path: str):
        audio_int16 = self.get_resampled_audio()
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            wf.setframerate(self.target_rate)
            wf.writeframes(audio_int16.tobytes())
        logger.info(f"Сохранено в WAV: {path}")


class FeaturesAudio:
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
            logger.warning("VAD-фичи ещё не реализованы")
            pass
