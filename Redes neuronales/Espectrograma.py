##### Comparación de señal original y sintetizada

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import librosa

## Base de datos PMA-Tidigit analizada con DNN y con red SincNet propuesta
# path_original = '..\\SingleWordProductionDutch-main\\PMA\\TiDigit\\LC\\wav\\TiDigit8May-6012193A.wav'
# path_sintetizada_DNN = '.\\Resultados_audio_PMA_TiDigit_DNN\\TiDigit8May-6012193A.wav'
# path_sintetizada_SincNet = '.\\Resultados_audio_PMA_TiDigit_SincNet\\TiDigit8May-6012193A.wav'

# rate_original, original = waves.read(path_original)
# rate_sintetizada_DNN, sintetizada_DNN = waves.read(path_sintetizada_DNN)
# rate_sintetizada_SincNet, sintetizada_SincNet = waves.read(path_sintetizada_SincNet)

###############

## Base de datos PMA-Arctic analizada con DNN y con red SincNet propuesta
# path_original = '..\\SingleWordProductionDutch-main\\PMA\\Arctic\\JG\\wav\\Arctic15Jul-a0001A.wav'
# path_sintetizada_DNN = '.\\Resultados_audio_PMA_Arctic_DNN\\Arctic15Jul-a0001A.wav'
# path_sintetizada_SincNet = '.\\Resultados_audio_PMA_Arctic_SincNet\\Arctic15Jul-a0001A.wav'

# rate_original, original = waves.read(path_original)
# rate_sintetizada_DNN, sintetizada_DNN = waves.read(path_sintetizada_DNN)
# rate_sintetizada_SincNet, sintetizada_SincNet = waves.read(path_sintetizada_SincNet)

###############

## Base de datos EEG analizada con red propuesta (sujeto M11)
# path_original = '..\\SingleWordProductionDutch-main\\HUVN\\M11\\ses01\\audio\\044_OLO_5.wav'
# path_sintetizada_DNN = '.\\Resultados_audio_EEG_DNN\\M11\\ses01\\001_IFI_1.wav'
# path_sintetizada_SincNet = '.\\Resultados_audio_EEG_SincNet\\M11\\ses01\\001_IFI_1.wav'

# rate_original, original = waves.read(path_original)
# rate_sintetizada_DNN, sintetizada_DNN = waves.read(path_sintetizada_DNN)
# rate_sintetizada_SincNet, sintetizada_SincNet = waves.read(path_sintetizada_SincNet)

###############

## Base de datos EEG analizada con red propuesta (sujeto F09)
path_original = '..\\SingleWordProductionDutch-main\\HUVN\\F09\\ses01\\audio\\001_OLO_1.wav'
path_sintetizada_DNN = '.\\Resultados_audio_EEG_DNN\\F09\\ses01\\001_OLO_1.wav'
path_sintetizada_SincNet = '.\\Resultados_audio_EEG_SincNet\\F09\\ses01\\001_OLO_1.wav'

rate_original, original = waves.read(path_original)
rate_sintetizada_DNN, sintetizada_DNN = waves.read(path_sintetizada_DNN)
rate_sintetizada_SincNet, sintetizada_SincNet = waves.read(path_sintetizada_SincNet)

###############


# Se ilustran la señal original y la sintetizada

tiempo_original = np.arange(0, len(original)/rate_original, 1/rate_original)
tiempo_sintetizada_DNN = np.arange(0, len(sintetizada_DNN)/rate_sintetizada_DNN, 1/rate_sintetizada_DNN)
tiempo_sintetizada_SincNet = np.arange(0, len(sintetizada_SincNet)/rate_sintetizada_SincNet, 1/rate_sintetizada_SincNet)

fig, axs = plt.subplots(3,2, figsize=(10, 8))

axs[0,0].plot(tiempo_original, original, color='blue')
axs[0,0].set_xlabel('Tiempo (segundos)')
axs[0,0].set_ylabel('Amplitud')
axs[0,0].set_title('Señal original')
axs[0,0].grid()

axs[1,0].plot(tiempo_sintetizada_DNN, sintetizada_DNN, color='red')
axs[1,0].set_xlabel('Tiempo (segundos)')
axs[1,0].set_ylabel('Amplitud')
axs[1,0].set_title('Señal sintetizada con DNN')
axs[1,0].grid()

axs[2,0].plot(tiempo_sintetizada_SincNet, sintetizada_SincNet, color='red')
axs[2,0].set_xlabel('Tiempo (segundos)')
axs[2,0].set_ylabel('Amplitud')
axs[2,0].set_title('Señal sintetizada con red SincNet propuesta')
axs[2,0].grid()

# Se obtienen los espectrogramas y se ilustran 

original, sr_orig = librosa.load(path_original)
original = librosa.stft(original, n_fft=2048, hop_length=512)
original_dB = librosa.amplitude_to_db(np.abs(original), ref=np.max)

librosa.display.specshow(original_dB, sr=sr_orig, x_axis='time', y_axis='hz', ax=axs[0,1])
axs[0,1].set_xlabel('tiempo (s)')
axs[0,1].set_ylabel('frecuencia (Hz)')
axs[0,1].set_title('Espectrograma señal original')
axs[0,1].grid()
axs[0,1].set_ylim(0,8000)
# plt.colorbar(format='%+2.0f dB', ax=axs[0,1])


sintetizada_DNN, sr_sint = librosa.load(path_sintetizada_DNN)
sintetizada_DNN = librosa.stft(sintetizada_DNN, n_fft=2048, hop_length=512)
sintetizada_DNN_dB = librosa.amplitude_to_db(np.abs(sintetizada_DNN), ref=np.max)

librosa.display.specshow(sintetizada_DNN_dB, sr=sr_sint, x_axis='time', y_axis='hz', ax=axs[1,1])
axs[1,1].set_xlabel('tiempo (s)')
axs[1,1].set_ylabel('frecuencia (Hz)')
axs[1,1].set_title('Espectrograma señal sintetizada con DNN')
axs[1,1].grid()
axs[1,1].set_ylim(0,8000)
# plt.colorbar(format='%+2.0f dB', ax=axs[1,1])


sintetizada_SincNet, sr_sint = librosa.load(path_sintetizada_SincNet)
sintetizada_SincNet = librosa.stft(sintetizada_SincNet, n_fft=2048, hop_length=512)
sintetizada_SincNet_dB = librosa.amplitude_to_db(np.abs(sintetizada_SincNet), ref=np.max)

librosa.display.specshow(sintetizada_SincNet_dB, sr=sr_sint, x_axis='time', y_axis='hz', ax=axs[2,1])
axs[2,1].set_xlabel('tiempo (s)')
axs[2,1].set_ylabel('frecuencia (Hz)')
axs[2,1].set_title('Espectrograma señal sintetizada con red SincNet')
axs[2,1].grid()
axs[2,1].set_ylim(0,8000)
# plt.colorbar(format='%+2.0f dB', ax=axs[1,1])

plt.tight_layout()
plt.show()