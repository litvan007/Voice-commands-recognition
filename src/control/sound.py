import time
from settings import Settings
import RPi.GPIO as GPIO

class SoundController:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.BUZZER_PIN = 16
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.BUZZER_PIN, GPIO.OUT)
        
    def play_sound(self, sound_type: str):
        """Воспроизводит звук указанного типа"""
        sound_config = getattr(self.settings.sounds, sound_type)
        
        # Включаем звук (50% скважность)
        pwm = GPIO.PWM(self.BUZZER_PIN, sound_config.frequency) 
        pwm.start(100)
        
        # Ждем указанное время
        time.sleep(sound_config.duration)
        
        # Выключаем звук
        pwm.stop()
        
    def play_start(self):
        """Воспроизводит звук при старте"""
        self.play_sound('start')
        
    def play_stop(self):
        """Воспроизводит звук при остановке"""
        self.play_sound('stop')
        
    def play_action_start(self):
        """Воспроизводит звук при начале действия"""
        self.play_sound('action_start')
        
    def play_action_end(self):
        """Воспроизводит звук при завершении действия"""
        self.play_sound('action_end')
        
    def play_recognition_failed(self):
        """Воспроизводит звук при неудачном распознавании"""
        self.play_sound('recognition_failed') 