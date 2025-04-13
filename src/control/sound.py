import time
from settings import Settings

class SoundController:
    def __init__(self, pca, settings: Settings):
        self.pca = pca
        self.settings = settings
        self.pca.frequency = 1000  # Базовая частота ШИМ
        
    def play_sound(self, sound_type: str):
        """Воспроизводит звук указанного типа"""
        sound_config = getattr(self.settings.sounds, sound_type)
        
        # Включаем звук (50% скважность)
        self.pca.channels[sound_config.channel].duty_cycle = 0x7FFF
        
        # Ждем указанное время
        time.sleep(sound_config.duration)
        
        # Выключаем звук
        self.pca.channels[sound_config.channel].duty_cycle = 0
        
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