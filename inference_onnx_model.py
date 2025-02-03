import onnxruntime as ort
import numpy as np

import scipy.io.wavfile as wav
import python_speech_features as psf


session = ort.InferenceSession('./model.onnx')
input_name = session.get_inputs()[0].name

" Выводим, что модель инициализируется "
print( session, input_name )


wav_path = '/home/i.litvinov/Voice-commands-recognition/user_102864961/Старт_1_1.wav'

sample_rate, signal = wav.read(wav_path)
if signal.ndim > 1:
    signal = signal[:, 0]

print( sample_rate, signal )

" Для вычисления MFCC "

winlen = 480 / sample_rate
winstep = 160 / sample_rate
n_mfcc = 32
n_filt = 64
nfft = 480

mfcc_feat = psf.mfcc(signal,
                     samplerate=sample_rate,
                     winlen=winlen,
                     winstep=winstep,
                     numcep=n_filt,
                     nfft=nfft)

print( "Размерность  MFCC", mfcc_feat.shape )

