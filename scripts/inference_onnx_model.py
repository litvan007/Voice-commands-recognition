import onnxruntime as ort
import numpy as np

import librosa
import librosa.display
import matplotlib.pyplot as plt

import scipy.io.wavfile as wav
import speechpy.feature as feature
from python_speech_features import mfcc

import time


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)

# Маппинг классов
labels_map = {  0: 'Опустить',
                1: 'Двигаться',
                2: 'Сохранить',
                3: 'Вниз',
                4: 'Старт',
                5: 'Открыть',
                6: 'Сменить',
                7: 'Вверх',
                8: 'Захватить',
                9: 'Влево',
                10: 'Поднять',
                11: 'Закрыть',
                12: 'Остановиться',
                13: 'Загрузить',
                14: 'Найти',
                15: 'Стоп',
                16: 'Домой',
                17: 'Вправо'}

if __name__ == '__main__':

    # инициализация модели
    session = ort.InferenceSession('./model.onnx')
    input_name = session.get_inputs()[0].name

    " Выводим, что модель инициализируется "
    print( session, input_name )


    # чтение через librosa (совпадает с torch)
    wav_path = '/home/i.litvinov/Voice-commands-recognition/output_new.wav'
    print( f'Wav name: {wav_path}' )
    # wav_path = '/home/i.litvinov/Voice-commands-recognition/user_102864961/Загрузить_12_1.wav'
    signal, sample_rate = librosa.load(wav_path, sr=16000)
    print( sample_rate, signal, signal.shape )

    (fs, signal) = wav.read(wav_path)

    if signal.dtype == np.int16: # can be deleted
        signal = signal.astype(np.float32) / 32768.0

    if signal.ndim > 1:
        signal = signal[:, 0]

    print( sample_rate, signal, signal.shape )

    " Для вычисления MFCC "

    sample_rate = 16000
    n_fft = 480
    hop_length = 160
    n_mels = 64
    n_mfcc = 32

    time_mfcc_start = time.time()

    # mfcc = mfcc(signal,
    #             sample_rate,
    #             winlen=0.03,
    #             winstep=0.01,
    #             numcep=32,
    #             nfilt=64,
    #             nfft=480).T

    # mfcc = feature.mfcc(
    #                     signal,
    #                     sampling_frequency=fs,
    #                     frame_length=0.03,
    #                     frame_stride=0.01,
    #                     num_cepstral=32,
    #                     num_filters=64,
    #                     fft_length=480
    # ).T


    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        dct_type=2,
        norm='ortho'
    )
    
    print( f'Время обработки MFCC признаков: {time.time() - time_mfcc_start}' )

    print( "Размерность  MFCC", mfcc.shape )

    print( mfcc )

    " Вычисления на модели "

    input_np = mfcc.T
    input_np = np.expand_dims(input_np, axis=0)
    input_np = np.expand_dims(input_np, axis=0)

    input_np = input_np.astype(np.float32)

    print( input_np.dtype, input_np.shape )
    time_model_start = time.time()
    result = session.run(None, {input_name: input_np})
    print( f'Common computing time: {time.time() - time_model_start}' )

    print( result )

    probs_result = softmax(result[0])
    print( probs_result )

    print( labels_map[np.argmax(probs_result)] )
