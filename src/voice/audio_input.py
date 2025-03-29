import librosa
import numpy as np
import logging
from scipy.fftpack import dct
from settings import Settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # можно сделать DEBUG при отладке

class FeaturesAudio:
    def sample(self, wave_path: str, settings: Settings):
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
