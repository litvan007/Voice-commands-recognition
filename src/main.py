import logging
import smbus
from settings import Settings
from voice.audio_input import AudioRecorder, FeaturesAudio, prepare_audio_device
from voice.recognizer import SpeechCommandModel
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
    logger.info("🚀 Запуск голосового управления")

    audio_extractor = FeaturesAudio()
    model = SpeechCommandModel(settings)

    # Инициализация I2C и руки
    i2c_bus = smbus.SMBus(1)
    pca = PCA9685(i2c_bus)
    arm = ArmController(pca)

    # Однократная инициализация аудиоустройства
    audio, device_index, device_info = prepare_audio_device(2)

    is_active = False

    while True:
        logger.info("🕓 Ожидание голосовой команды...")
        recorder = AudioRecorder(
            audio=audio,
            device_index=device_index,
            device_info=device_info,
            sample_rate=settings.audio_config.common.sample_rate
        )

        recorder.frames.clear()
        recorder.stop_recording = False
        recorder.start_recording()
        signal = recorder.get_resampled_audio()

        mfcc = audio_extractor.get_features(signal, settings, feature_type="VCR")
        label, confidence = model.predict(mfcc)

        if label is None:
            logger.warning(f"Команда отвергнута: низкая уверенность (p={confidence:.2f})")
            continue

        logger.info(f"✅ Распознана команда: {label} (p={confidence:.2f})")

        if label.lower() == "старт":
            is_active = True
            logger.info("🎬 Система активирована")
            arm.load_servo_positions()

        elif label.lower() == "стоп":
            is_active = False
            logger.info("🛑 Система остановлена")
            arm.save_servo_positions()

        elif is_active and label.lower() == "остановиться":
            logger.info("⏹ Экстренное прерывание")
            arm.disable_all_servos()
            arm.save_servo_positions()

        elif is_active:
            arm.execute_command(label, settings)

        else:
            logger.info("⚠️ Система неактивна. Произнесите «Старт» для начала.")


if __name__ == '__main__':
    main()
