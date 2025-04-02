import logging
import time
from settings import Settings
from voice.audio_input import FeaturesAudio, AudioRecorder
from voice.recognizer import SpeechCommandModel
from control.controller import PCA9685, ArmController

def main():
    settings = Settings.load()
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Запуск потоковой голосовой системы")

    audio_processor = FeaturesAudio()
    recorder = AudioRecorder(device_index=2, sample_rate=16000)
    model = SpeechCommandModel(settings)
    arm = ArmController(PCA9685(smbus.SMBus(1)))

    listening_mode = False  # Изначально система не активна

    while True:
        logger.info("Ожидание голосовой команды...")
        recorder.start_recording()
        signal = recorder.get_resampled_audio()
        mfcc = audio_processor.get_features(signal, settings, "VCR")
        label, confidence = model.predict(mfcc)

        if not label or confidence < 0.6:
            logger.info("Команда не распознана или низкая уверенность.")
            continue

        logger.info(f"Распознано: {label} (уверенность {confidence:.2f})")

        if label == "Старт":
            listening_mode = True
            logger.info("🟢 Распознавание команд активировано.")
            arm.execute_command(label, settings)  # Можно и двигаться в позицию «Старт»

        elif label == "Стоп":
            listening_mode = False
            logger.info("🔴 Распознавание команд остановлено.")
            arm.execute_command(label, settings)  # Вернуться в нейтральную позицию

        elif listening_mode:
            logger.info(f"Выполняем команду: {label}")
            arm.execute_command(label, settings)

        else:
            logger.info(f"Система в режиме ожидания. Команда '{label}' проигнорирована.")

        time.sleep(0.5)

if __name__ == "__main__":
    main()
