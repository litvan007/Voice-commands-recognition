from pydantic import BaseModel, Field
from typing import Dict, Optional
import yaml
from pathlib import Path
# Пути к конфигам
AUDIO_CONFIG_PATH = Path('/Users/litvan007/Voice-commands-recognition/configs/audio_config.yaml')
VOICE_COMMAND_CONFIG_PATH = Path('/Users/litvan007/Voice-commands-recognition/configs/voice_commands.yaml')
CONTROL_CONFIG_PATH = Path('...')  # TODO

MODELS_CONFIG_PATH = Path('/Users/litvan007/Voice-commands-recognition/configs/models.yaml')
ARM_COMMAND_CONFIG_PATH = Path('/Users/litvan007/Voice-commands-recognition/configs/arm_commands.yaml')

class ArmCommandSet(BaseModel):
    commands: Dict[str, Dict[int, int]]  # label -> {channel: pulse}

# --- AUDIO CONFIG ---
class CommonAudioConfig(BaseModel):
    sample_rate: int
    debug: bool

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
    frame_duration_ms: int
    threshold: float
    smoothing_window: int
    min_speech_duration_ms: int
    min_silence_duration_ms: int

class AudioConfig(BaseModel):
    common: CommonAudioConfig
    command_features: CommandFeatures
    vad_features: VADFeatures

# --- VOICE COMMAND CONFIG ---

class ConfidenceThresholds(BaseModel):
    default: float
    overrides: Dict[int, float]

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

# --- SETTINGS CLASS ---

class Settings(BaseModel):
    arm_commands: ArmCommandSet
    audio_config: AudioConfig
    voice_command_config: VoiceCommandConfig
    model_paths: ModelPaths
    debug: bool

    @classmethod
    def load(cls) -> "Settings":
        with open(ARM_COMMAND_CONFIG_PATH, "r", encoding="utf-8") as f:
            arm_data = yaml.safe_load(f)
        with open(AUDIO_CONFIG_PATH, "r", encoding="utf-8") as f:
            audio_data = yaml.safe_load(f)
        with open(VOICE_COMMAND_CONFIG_PATH, "r", encoding="utf-8") as f:
            vc_data = yaml.safe_load(f)
        with open(MODELS_CONFIG_PATH, "r", encoding="utf-8") as f:
            model_data = yaml.safe_load(f)

        return cls(
            audio_config=AudioConfig(**audio_data),
            voice_command_config=VoiceCommandConfig(**vc_data),
            model_paths=ModelPaths(**model_data["models"]),
            arm_commands=ArmCommandSet(**arm_data),
            debug=AudioConfig(**audio_data).common.debug
        )

