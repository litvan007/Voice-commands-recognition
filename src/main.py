import logging
from settings import Settings
from voice.audio_input import AudioRecorder, FeaturesAudio
from voice.recognizer import SpeechCommandModel
from control.controller import PCA9685, ArmController
import smbus

def setup_logging(debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
    )

def main():
    settings = Settings.load()
    setup_logging(settings.debug)
    logger = logging.getLogger(__name__)
    logger.info("Запуск системы голосового управления")

    audio = FeaturesAudio()
    model = SpeechCommandModel(settings)

    i2cBus = smbus.SMBus(1)
    pca9685 = PCA9685(i2cBus)
    arm = ArmController(pca9685)

    is_active = False  # Изначально система неактивна

    recorder = AudioRecorder(
        device_index=2,
        sample_rate=settings.audio_config.common.sample_rate
    )
    recorder.list_devices()

    while True:
        logger.info("Ожидание голосовой команды...")
        recorder.frames.clear()
        recorder.stop_recording = False
        recorder.start_recording()
        signal = recorder.get_resampled_audio()

        mfcc = audio.get_features(signal, settings, feature_type="VCR")
        label, confidence = model.predict(mfcc)

        if label is None:
            logger.warning(f"Команда отвергнута: низкая уверенность (p={confidence:.2f})")
            continue

        logger.info(f"Распознана команда: {label} (p={confidence:.2f})")

        if label.lower() == "старт":
            is_active = True
            logger.info("✅ Система активирована. Можно отдавать команды.")
            arm.execute_command(label, settings)  # опционально дать руке позицию "Старт"

        elif label.lower() == "стоп":
            is_active = False
            logger.info("⛔️ Система остановлена. Команды не принимаются.")
            arm.execute_command(label, settings)  # опционально нейтральное положение

        elif is_active:
            arm.execute_command(label, settings)
        else:
            logger.info("⚠️ Система неактивна. Произнесите команду «Старт» для начала работы.")

if __name__ == '__main__':
    main()
