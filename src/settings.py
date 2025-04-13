from pydantic import BaseModel, Field
from typing import Dict, Optional
import yaml
from pathlib import Path
# Пути к конфигам
AUDIO_CONFIG_PATH = Path('/home/i.litvinov/Voice-commands-recognition/configs/audio_config.yaml')
VOICE_COMMAND_CONFIG_PATH = Path('/home/i.litvinov/Voice-commands-recognition/configs/voice_commands.yaml')
CONTROL_CONFIG_PATH = Path('...')  # TODO
SOUNDS_CONFIG_PATH = Path('/home/i.litvinov/Voice-commands-recognition/configs/sounds.yaml')

MODELS_CONFIG_PATH = Path('/home/i.litvinov/Voice-commands-recognition/configs/models.yaml')
ARM_COMMAND_CONFIG_PATH = Path('/home/i.litvinov/Voice-commands-recognition/configs/arm_commands.yaml')

class ArmCommandSet(BaseModel):
    commands: Dict[str, Dict[int, int]]  # label -> {channel: pulse}

# --- AUDIO CONFIG ---
class CommonAudioConfig(BaseModel):
    sample_rate: int
    debug: bool
    desired_length: int

class CommandFeatures(BaseModel):
    n_fft: int
    hop_length: int
    n_mels: int
    n_mfcc: int
    window: Optional[str] = "hann"
    normalize: Optional[bool] = True
    center: Optional[bool] = True
    power: Optional[float] = 2.0
    fmin: Optional[int] = 0
    fmax: Optional[int] = None
    htk: Optional[bool] = False

class VADFeatures(BaseModel):
    merge_segments: bool
    merge_gap: float   
    min_signal_lenght: int

class RingBuffer(BaseModel):
    duration: int
    update_interval: int
    use_ring_buffer: bool

class AudioConfig(BaseModel):
    common: CommonAudioConfig
    command_features: CommandFeatures
    vad_features: VADFeatures
    ring_buffer: RingBuffer

# --- VOICE COMMAND CONFIG ---

class ConfidenceThresholds(BaseModel):
    vcr_default: float
    vcr_overrides: Dict[int, float]
    vad_default: float

class VoiceCommandConfig(BaseModel):
    labels_map: Dict[int, str]
    confidence_thresholds: ConfidenceThresholds


# --- MODELS INIT CONFIG ---
class ModelPaths(BaseModel):
    vad: str
    command_recognizer: str
    command_recognizer_large: Optional[str] = None

class ModelConfig(BaseModel):
    models: ModelPaths

# --- SOUNDS CONFIG ---

class SoundConfig(BaseModel):
    frequency: int
    duration: float
    channel: int

class SoundsConfig(BaseModel):
    start: SoundConfig
    stop: SoundConfig
    action_start: SoundConfig
    action_end: SoundConfig
    recognition_failed: SoundConfig

# --- SETTINGS CLASS ---

class Settings(BaseModel):
    arm_commands: ArmCommandSet
    audio_config: AudioConfig
    voice_command_config: VoiceCommandConfig
    model_paths: ModelPaths
    sounds: SoundsConfig
    debug: bool

    @classmethod
    def load(cls) -> "Settings":
        with open(AUDIO_CONFIG_PATH) as f:
            audio_config = AudioConfig(**yaml.safe_load(f))
            
        with open(VOICE_COMMAND_CONFIG_PATH) as f:
            voice_command_config = VoiceCommandConfig(**yaml.safe_load(f))
            
        with open(ARM_COMMAND_CONFIG_PATH) as f:
            arm_commands = ArmCommandSet(**yaml.safe_load(f))
            
        with open(MODELS_CONFIG_PATH) as f:
            model_paths = ModelPaths(**yaml.safe_load(f)['models'])
            
        with open(SOUNDS_CONFIG_PATH) as f:
            sounds = SoundsConfig(**yaml.safe_load(f)['sounds'])

        return cls(
            arm_commands=arm_commands,
            audio_config=audio_config,
            voice_command_config=voice_command_config,
            model_paths=model_paths,
            sounds=sounds,
            debug=audio_config.common.debug
        )

