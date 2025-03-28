import librosa
import numpy as np
from scipy.fftpack import dct


class FeaturesAudio:
    def __init__(self, sample_rate) -> None:
        self.sample_rate = sample_rate
        pass

    def sample(self, wave_path: str):
        signal, sample_rate = librosa.load(wave_path, sr=self.sample_rate)
    
        if signal.dtype == np.int16: # can be deleted
            signal = signal.astype(np.float32) / 32768.0

        if signal.ndim > 1:
            signal = signal[:, 0]

        return signal, sample_rate

    def get_features(self, signal, sample_rate, config, feature_type: str): #TODO
        if feature_type == 'VCR':

            # STFT -> Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram( #TODO
                y=signal, sr=sample_rate,
                n_fft=480, hop_length=160, win_length=480,
                window='hann', center=True, pad_mode='reflect',
                power=2.0,         # power spectrogram (mag^2)
                n_mels=64, 
                fmin=0.0, fmax=8000.0,
                htk=True, norm=None  # mimic Torchaudio: HTK mel, no filter norm&#8203;:contentReference[oaicite:12]{index=12}
            )

            # dB conversion (10 * log10), reference max, no clipping
            # Avoid log10(0) by adding a tiny value (Librosa uses amin=1e-10 for power) 
            amin = 1e-10
            mel_spec = np.maximum(mel_spec, amin)
            ref_value = mel_spec.max()
            mel_spec_db = 10.0 * np.log10(mel_spec / ref_value)  # 0 dB at max&#8203;:contentReference[oaicite:13]{index=13}

            # DCT-II along the mel axis to get MFCCs
            mfcc = dct(mel_spec_db, type=2, axis=0, norm='ortho')[0:32, :]

            return mfcc


        if feature_type == 'VAD':
            pass
