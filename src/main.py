import logging
import smbus
from settings import Settings
from voice.audio_input import AudioRecorder, RingBufferAudioRecorder, FeaturesAudio, prepare_audio_device
from voice.recognizer import SpeechCommandModel
from voice.vad import EmobaseCNN
from control.controller import PCA9685, ArmController
from utils import estimate_snr, plot_vad_segments
import numpy as np
import time

def setup_logging(debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
    )


def process_command(signal, valid_frames, settings, audio_extractor, vad, model, arm, is_active):
    """Обработка команды из аудиосигнала"""
    logger = logging.getLogger(__name__)
    
    if settings.debug:
        snr = estimate_snr(signal)
        logger.debug(f"SNR: {snr:.2f} dB")


    # -- Segmentation -- 
    vad_features = audio_extractor.get_features(signal, settings, feature_type="VAD")
    vad_mask = vad.predict(vad_features)
    segments = vad.segment(vad_mask, valid_frames)

    print( segments )

    if len(segments) > 0:
        logger.info(f"VAD модель обнаружила голосовую активность. Отсчетов: {segments}")
        speech_signal = np.concatenate([signal[start:end] for start, end in segments])
        
        if len(speech_signal) < settings.audio_config.vad_features.min_signal_lenght:
            logger.info("📉 Слишком короткий речевой сегмент — пропуск")
            return is_active, False

        if settings.debug:
            logger.debug("Отрисовка графика с границами")
            plot_vad_segments(signal, settings.audio_config.common.sample_rate, segments)

        # -- Recognition --
        mfcc = audio_extractor.get_features(speech_signal, settings, feature_type="VCR")
        label, confidence = model.predict(mfcc)

        if label is None:
            logger.warning(f"Команда отвергнута: низкая уверенность (p={confidence:.2f})")
            return is_active, False

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

        return is_active, True

    return is_active, False


def main():
    settings = Settings.load()
    setup_logging(settings.debug)
    logger = logging.getLogger(__name__)
    logger.info("🚀 Запуск голосового управления")

    audio_extractor = FeaturesAudio()
    vad = EmobaseCNN(settings)
    model = SpeechCommandModel(settings)

    # Инициализация I2C и руки
    i2c_bus = smbus.SMBus(1)
    pca = PCA9685(i2c_bus)
    arm = ArmController(pca)

    # Однократная инициализация аудиоустройства
    audio, device_index, device_info = prepare_audio_device(3)

    is_active = False

    if settings.audio_config.ring_buffer.use_ring_buffer:
        # Используем кольцевой буфер
        recorder = RingBufferAudioRecorder(
            audio=audio,
            device_index=device_index,
            device_info=device_info,
            sample_rate=settings.audio_config.common.sample_rate,
            buffer_duration=settings.audio_config.ring_buffer.duration,
            update_interval=settings.audio_config.ring_buffer.update_interval
        )
        recorder.start_recording()

        try:
            while True:
                # Проверяем, нужно ли запускать VAD
                if recorder.should_check_vad():
                    logger.info("🕓 Проверка голосовой активности...")
                    
                    # Получаем текущее содержимое буфера
                    signal, valid_frames = recorder.get_resampled_audio()
                    
                    is_active, command_processed = process_command(
                        signal, valid_frames, settings, audio_extractor, vad, model, arm, is_active
                    )
                    
                    if command_processed:
                        recorder.clear_buffer()

        except KeyboardInterrupt:
            logger.info("👋 Завершение работы...")
        finally:
            recorder.stop_recording()
    else:
        # Используем обычную запись
        while True:
            logger.info("🕓 Ожидание голосовой команды...")
            recorder = AudioRecorder(
                audio=audio,
                device_index=device_index,
                device_info=device_info,
                sample_rate=settings.audio_config.common.sample_rate,
                desired_length=settings.audio_config.common.desired_length
            )

            recorder.frames.clear()
            recorder.stop_recording = False
            recorder.start_recording()
            signal, valid_frames = recorder.get_resampled_audio()

            if settings.debug:
                recorder.save_to_wav(signal)

            is_active, _ = process_command(
                signal, valid_frames, settings, audio_extractor, vad, model, arm, is_active
            )


if __name__ == '__main__':
    main()
