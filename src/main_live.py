import logging
import time
import numpy as np
from settings import Settings
from voice.audio_input import FeaturesAudio
from voice.recognizer import SpeechCommandModel
from voice.recorder import AudioRecorder
from control.controller import PCA9685, ArmController

def setup_logging(debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
    )

def main():
    settings = Settings.load()
    setup_logging(settings.debug)
    logger = logging.getLogger(__name__)
    logger.info("🔁 Голосовой режим (потоковая работа) запущен")

    # Компоненты
    audio = FeaturesAudio()
    model = SpeechCommandModel(settings)
    driver = PCA9685()
    arm = ArmController(driver)

    # Микрофон
    recorder = AudioRecorder(
        device_index=2,
        sample_rate=settings.audio_config.common.sample_rate
    )
    recorder.list_devices()

    logger.info("🎤 Начинаем потоковую запись...")

    try:
        while True:
            # 1. Записать короткий фрагмент (например, 2 секунды)
            recorder.start_recording()
            time.sleep(2)
            recorder.stop_recording = True
            signal = recorder.get_resampled_audio()

            if len(signal) < settings.audio_config.common.sample_rate // 2:
                logger.debug("🔇 Слишком короткий сигнал, пропускаем")
                continue

            # 2. Извлечь признаки и предсказать
            mfcc = audio.get_features(signal, settings, feature_type="VCR")
            label, confidence = model.predict(mfcc)

            if label:
                logger.info(f"✅ Распознано: {label} (p={confidence:.2f})")
                arm.execute_command(label, settings)
            else:
                logger.debug("❌ Команда не распознана")

    except KeyboardInterrupt:
        logger.info("⏹ Работа остановлена пользователем")

if __name__ == "__main__":
    main()
