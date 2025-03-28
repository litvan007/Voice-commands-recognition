import numpy as np
import scipy.fftpack
import librosa

def compute_mfcc(signal, sample_rate=16000, n_fft=480, hop_length=160,
                 n_mels=64, n_mfcc=32, eps=1e-10):
    # Если сигнал – одномерный массив (моно), можно работать напрямую.
    # Если center=True (как в torchaudio), то дополните сигнал:
    pad = n_fft // 2
    signal = np.pad(signal, pad_width=pad, mode='reflect')
    
    # Вычисляем STFT. Здесь мы задаём win_length равным n_fft и используем Hann-окно.
    S_complex = librosa.stft(signal,
                             n_fft=n_fft,
                             hop_length=hop_length,
                             win_length=n_fft,
                             window='hann',
                             center=False)  # center уже учтён ручным паддингом
    S_power = np.abs(S_complex) ** 2

    # Создаём мел-фильтрбанк. Обратите внимание, что здесь используются стандартные настройки.
    mel_filter = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_filter, S_power)
    
    # Логарифмическое масштабирование
    log_mel_spec = np.log(mel_spec + eps)
    
    # Применяем DCT (тип II) вдоль оси мел-фильтров (по оси 0), и берем первые n_mfcc коэффициентов.
    # Обратите внимание: torchaudio использует нормировку 'ortho'
    mfcc = scipy.fftpack.dct(log_mel_spec, type=2, axis=0, norm='ortho')[:n_mfcc, :]
    return mfcc

# Пример использования:
import scipy.io.wavfile as wav

sample_rate, signal = wav.read("your_file.wav")
# Если стерео, берём первый канал:
if signal.ndim > 1:
    signal = signal[:, 0]
# Если сигнал в int16, нормализуем:
if signal.dtype != np.float32:
    signal = signal.astype(np.float32) / np.iinfo(signal.dtype).max

mfcc = compute_mfcc(signal, sample_rate=sample_rate, n_fft=480, hop_length=160,
                    n_mels=64, n_mfcc=32)
print(mfcc.shape)  # Обычно (32, T) где T – число фреймов
