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

def plot_vad_segments(signal: np.ndarray, sample_rate: int, segments: list, command_name: str = None):
    """
    Отрисовывает график сигнала с выделенными сегментами голосовой активности.
    
    Args:
        signal: Аудиосигнал
        sample_rate: Частота дискретизации
        segments: Список сегментов в формате [(start, end), ...]
        command_name: Название распознанной команды (опционально)
    """
    # Создаем временную ось
    time = np.arange(len(signal)) / sample_rate
    
    # Создаем график
    plt.figure(figsize=(12, 4))
    
    # Рисуем сигнал
    plt.plot(time, signal, alpha=0.5, label='Сигнал')
    
    # Выделяем сегменты голосовой активности
    for start, end in segments:
        plt.axvspan(start/sample_rate, end/sample_rate, 
                   color='red', alpha=0.3, label='Голосовая активность')
    
    # Добавляем название команды в заголовок, если оно предоставлено
    title = "Сегменты голосовой активности"
    if command_name:
        title += f" - Команда: {command_name}"
    
    plt.title(title)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    
    # Убираем дублирующиеся легенды
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
