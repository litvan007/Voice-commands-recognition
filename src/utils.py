import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from typing import Optional

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

def plot_vad_segments(signal: np.ndarray, sample_rate: int, segments: np.ndarray, command_name: Optional[str] = None):
    """
    Визуализирует аудиосигнал с выделенными сегментами речи.
    
    Args:
        signal: Аудиосигнал
        sample_rate: Частота дискретизации
        segments: Массив сегментов в формате [[start1, end1], [start2, end2], ...]
        command_name: Название распознанной команды (опционально)
    """
    plt.figure(figsize=(15, 5))
    
    # Создаем временную ось
    time = np.linspace(0, len(signal) / sample_rate, len(signal))
    
    # Рисуем сигнал
    plt.plot(time, signal, alpha=0.5, label='Аудиосигнал')
    
    # Выделяем сегменты речи
    for start, end in segments:
        plt.axvspan(start/sample_rate, end/sample_rate, color='red', alpha=0.3)
    
    # Добавляем название команды в заголовок, если оно предоставлено
    title = "Распознанная команда: " + command_name if command_name else "Границы речевых сегментов"
    plt.title(title)
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.legend()
    
    # Создаем директорию для графиков, если её нет
    os.makedirs('debug_plots', exist_ok=True)
    
    # Сохраняем график в файл
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join('debug_plots', f"vad_segments_{timestamp}.png")
    plt.savefig(filename)
    plt.close()
    
    logger.debug(f"График сохранен в файл: {filename}")
