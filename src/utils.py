import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def estimate_snr(signal: np.ndarray, sr: int = 16000, noise_sec: float = 0.5) -> float:
    """
    Примерная оценка SNR из аудио: считаем начальные и конечные 0.5 сек — шумом.
    """
    n_noise = int(noise_sec * sr)

    noise = np.concatenate([signal[:n_noise], signal[-n_noise:]])
    signal_body = signal[n_noise:-n_noise] if len(signal) > 2 * n_noise else signal

    power_noise = np.mean(noise ** 2)
    power_signal = np.mean(signal_body ** 2)

    if power_noise == 0:
        return float("inf")

    snr_db = 10 * np.log10(power_signal / power_noise)
    return snr_db


