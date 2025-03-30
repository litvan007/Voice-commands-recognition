import logging
import librosa
from settings import Settings
from voice.audio_input import AudioRecorder, FeaturesAudio
from voice.recognizer import SpeechCommandModel
from control.controller import PCA9685, ArmController
import scipy.io.wavfile as wav
import time
import numpy as np

def setup_logging(debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
    )

def main():
    # 1. Загрузка конфигурации
    settings = Settings.load()
    setup_logging(settings.debug)
    logger = logging.getLogger(__name__)
    logger.info("Запуск системы голосового управления")

    # 2. Инициализация компонентов
    audio = FeaturesAudio()
    model = SpeechCommandModel(settings)
    driver = PCA9685()
    arm = ArmController(driver)

    # # 3. Запись с микрофона
    recorder = AudioRecorder(
        device_index=2,
        sample_rate=settings.audio_config.common.sample_rate
    )
    recorder.list_devices()
    recorder.start_recording()
    signal = recorder.get_resampled_audio()

    # 4. Извлечение признаков
    mfcc = audio.get_features(signal, settings, feature_type="VCR")

    # 5. Предсказание команды
    label, confidence = model.predict(mfcc)

    if label:
        logger.info(f"Выполнение команды: {label} (p={confidence:.2f})")
        arm.execute_command(label, settings)
    else:
        logger.warning(f"Команда отвергнута: низкая уверенность (p={confidence:.2f})")

if __name__ == '__main__':
    
    main()
    