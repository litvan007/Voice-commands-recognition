import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

def plot_vad_segments(signal: np.ndarray, sample_rate: int, segments: np.ndarray, directory: str = 'debug_plots') -> None:
    """
    Отрисовывает график аудио сигнала с отмеченными сегментами голосовой активности.
    
    Аргументы:
        signal (np.ndarray): Аудио сигнал.
        sample_rate (int): Частота дискретизации аудио сигнала.
        segments (np.ndarray): Массив сегментов речи в формате [[start, end], ...],
                               где start и end заданы в отсчетах.
    """

    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"segmented_{timestamp}.png"
    path = os.path.join(directory, filename)

    # Формируем временную ось в секундах
    time_axis = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, signal, label="Аудио сигнал")
    
    # Отмечаем на графике интервалы речи, переводя отсчёты в секунды
    for start, end in segments:
        plt.axvspan(start / sample_rate, end / sample_rate, color='red', alpha=0.3, label='Речь')
    
    # Избавляемся от дублирования меток в легенде
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.title("Голосовая активность")
    plt.grid(True)
    # plt.show()
    plt.savefig(path)
