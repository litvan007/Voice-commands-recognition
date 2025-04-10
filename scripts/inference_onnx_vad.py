import sys

sys.path.append('/Users/litvan007/Voice-commands-recognition/src')

from settings import Settings
from voice.audio_input import  FeaturesAudio, prepare_audio_device
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

    wav_path = '/Users/litvan007/Voice-commands-recognition/output.wav'
    print( f'Wav name: {wav_path}' )
    # wav_path = '/home/i.litvinov/Voice-commands-recognition/user_102864961/Загрузить_12_1.wav'
    signal, sample_rate = librosa.load(wav_path, sr=16000)
    print( sample_rate, signal, signal.shape )

    (fs, signal) = wav.read(wav_path)

    if signal.dtype == np.int16: # can be deleted
        signal = signal.astype(np.float32) / 32768.0

    if signal.ndim > 1:
        signal = signal[:, 0]

    print( sample_rate, signal, signal.shape )

    vad_features = audio_extractor.get_features(signal, settings, feature_type="VAD")
    vad_mask = vad.predict(vad_features)
    segments = vad.segment(vad_mask, len(signal), settings.audio_config.common.sample_rate) #TODO

    print( segments )

    speech_signal = np.concatenate([signal[start:end] for start, end in segments])
    if len(speech_signal) < 2000:  # опциональный фильтр по длине
        logger.info("📉 Слишком короткий речевой сегмент — пропуск")
        # continue

    logger.info("Отрисовка графика с границами")
    sound_show(signal, f'{wav_path.split("/")[-1]}', vad_mask, vad_mask)


if __name__ == '__main__':
    main()
