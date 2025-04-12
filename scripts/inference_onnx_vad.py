import sys

sys.path.append('/Users/litvan007/Voice-commands-recognition/src')

from settings import Settings
from voice.audio_input import FeaturesAudio
from voice.vad import EmobaseCNN
from utils import *

import librosa
import scipy.io.wavfile as wav
import numpy as np

import logging

def setup_logging(debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
    )

def main():
    settings = Settings.load()
    setup_logging(settings.debug)
    logger = logging.getLogger(__name__)
    logger.info("Старт скрипта")

    audio_extractor = FeaturesAudio()
    vad = EmobaseCNN(settings)

    wav_path = '/Users/litvan007/Voice-commands-recognition/notebooks/recognized_20250403_211519.wav'
    print( f'Wav name: {wav_path}' )
    # wav_path = '/home/i.litvinov/Voice-commands-recognition/user_102864961/Загрузить_12_1.wav'
    original_signal, sample_rate = librosa.load(wav_path, sr=16000)
    # print( sample_rate, max(original_signal), min(original_signal), original_signal.shape )

    if original_signal.dtype == np.int16: # can be deleted
        original_signal = original_signal.astype(np.float32) / 32768.0

    if original_signal.ndim > 1:
        original_signal = original_signal[:, 0]

    # print( sample_rate, max(original_signal), min(original_signal), original_signal.shape )
    valid_frames = original_signal.shape[0]
    # print( valid_frames )
    if valid_frames < settings.audio_config.common.desired_length:
        pad_length = settings.audio_config.common.desired_length- valid_frames

        first_five = original_signal[:100]
        pad_segment = np.random.choice(first_five, size=pad_length, replace=True)
        padded_signal = np.concatenate([original_signal, pad_segment])
        signal = padded_signal
    
    else:
        signal = original_signal

    # print( sample_rate, signal, signal.shape )
    # print( signal.shape )

    vad_features = audio_extractor.get_features(signal, settings, feature_type="VAD")
    vad_mask = vad.predict(vad_features)
    segments = vad.segment(vad_mask, valid_frames)

    logger.info(f"VAD модель успешно отработала. Длина отсчетов голосовой активности: {segments}")


    speech_signal = np.concatenate([signal[start:end] for start, end in segments])
    if len(speech_signal) < settings.audio_config.vad_features.min_signal_lenght:  # опциональный фильтр по длине
        logger.info("📉 Слишком короткий речевой сегмент — пропуск")
        # continue

    logger.info("Отрисовка графика с границами")
    # В main.py, после получения сегментов:

    # Допустим, сигнал и sample_rate уже определены, а segments получены от vad.segment(...)
    plot_vad_segments(signal, settings.audio_config.common.sample_rate, segments)



if __name__ == '__main__':
    main()
