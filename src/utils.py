import numpy as np
import matplotlib.pyplot as plt

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

def sound_show(inp, title=None, marks_true=None, marks_pred=None, sample_rate=16000):
    """
    Отображает звуковой сигнал и отмечает интервалы (например, речь),
    полученные из marks_true и marks_pred.
    
    inp         : numpy-массив звукового сигнала
    title       : заголовок графика
    marks_true  : метки истинной разметки (torch.Tensor или numpy-массив)
    marks_pred  : метки предсказанной разметки (torch.Tensor или numpy-массив)
    sample_rate : частота дискретизации сигнала
    """
    time = np.linspace(0., inp.size / sample_rate, inp.size)
    plt.figure(figsize=(16, 6))
    signal_line, = plt.plot(time, inp, label='Signal')
    plt.title(title)
    
    handles = [signal_line]
    labels = ['Signal']
    
    ax = plt.gca()
    
    if marks_pred is not None:
        # Преобразуем marks_pred в numpy-массив, если нужно
        words_grid_pred = find_words_edges(marks_pred, sample_rate)
        for edges in words_grid_pred:
            span = ax.axvspan(edges[0], edges[1], color='green', alpha=0.2)
        handles.append(span)
        labels.append('Predicted speech')
    
    if marks_true is not None:
        # Аналогично для marks_true
        words_grid_true = find_words_edges(marks_true, sample_rate)
        for edges in words_grid_true:
            span = ax.axvspan(edges[0], edges[1], color='red', alpha=0.3, linestyle='--')
        handles.append(span)
        labels.append('True speech')
    
    plt.legend(handles, labels)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def find_words_edges(marks, sample_rate=16000):
    """
    Определяет интервалы, где распознана речь (значение метки == 1).
    marks : numpy-массив с бинарными значениями (0/1)
    Возвращает массив интервалов [начало, конец] в секундах.
    """
    lst = []
    in_speech = False
    temp = []
    for i, m in enumerate(marks):
        if m == 1 and not in_speech:
            temp = [i / sample_rate]
            in_speech = True
        elif m == 0 and in_speech:
            temp.append(i / sample_rate)
            lst.append(temp)
            in_speech = False
    if in_speech:
        temp.append(len(marks) / sample_rate)
        lst.append(temp)
    return np.array(lst)