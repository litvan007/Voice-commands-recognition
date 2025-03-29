import logging

from settings import Settings
from voice.audio_input import FeaturesAudio

settings = Settings.load()

settings = Settings.load()
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


audio = FeaturesAudio(settings)

signal, sr = audio.sample("data/some.wav")
mfcc = audio.get_features(signal, feature_type='VCR')
