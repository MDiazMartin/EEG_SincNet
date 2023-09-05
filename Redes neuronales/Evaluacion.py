##### Evaluación de los resultados obtenidos #####

import numpy as np
import os
import glob
import scipy.io.wavfile as waves
import matplotlib.pyplot as plt

from pystoi import stoi # Métrica STOI

##########################################################################################

def melCepstralDistance(x, y, discard_c0=True):
    '''
    Calcula la medidal Mel-Cepstral Distortion (MCD) entre dos matrices de MFCCs calculados con World.
    
    Las matrices deben tener el mismo número de coeficientes MFCC (axis=1). Si tienen distinto número de tramas (axis=0),
    se calcula la MCD para las primeras tramas comunes frames= min(x.shape[0], y.shape[0]).
    
    Parameters
    ----------
    x: matriz de MFCCs calculados con PyWorld de tamaño (frames_x, num_mfccs)
    
    y: matriz de MFCCs calculados con PyWorld de tamaño (frames_y, num_mfccs)
    
    Returns
    -------
    mcd: vector de tamaño min(frames_x, framex_y) con el MCD calculado para cada trama
    '''
    
    log_spec_dB_const = (10.0 * np.sqrt(2.0)) / np.log(10.0) 
    
    frames = min(x.shape[0], y.shape[0])
    idx = 1 if discard_c0 else 0
    
    diff = x[:frames,idx:] - y[:frames,idx:]
    return np.mean(log_spec_dB_const * np.linalg.norm(diff, axis=1))

##########################################################################################

# ## Para síntesis de los datos de PMA-TiDigit con SincNet ##
# sub = ['LC', 'TP'] # Dos pacientes (TiDigit)
# path_audios_orig = '..\\SingleWordProductionDutch-main\\PMA\\TiDigit'
# path_audios_sint = '.\\Resultados_audio_PMA_TiDigit_SincNet'
# output_folder_mfcc = '.\\Resultados_mfcc_PMA_TiDigit_SincNet\\'
# print('Resultados para TiDigit-SincNet')
# signal = 'TiDigit'

# En caso de usar una DNN
# sub = ['LC', 'TP'] # Dos pacientes (TiDigit)
# path_audios_orig = '..\\SingleWordProductionDutch-main\\PMA\\TiDigit'
# path_audios_sint = '.\\Resultados_audio_PMA_TiDigit_DNN'
# output_folder_mfcc = '.\\Resultados_mfcc_PMA_TiDigit_DNN\\'
# print('Resultados para TiDigit-DNN')
# signal = 'TiDigit'


## Para síntesis de los datos de PMA-Artic con SincNet ##
# sub = ['JG', 'RM'] # Dos pacientes (Arctic)
# path_audios_orig = '..\\SingleWordProductionDutch-main\\PMA\\Arctic'
# path_audios_sint = '.\\Resultados_audio_PMA_Arctic_SincNet'
# output_folder_mfcc = '.\\Resultados_mfcc_PMA_Arctic_SincNet\\'
# print('Resultados para Arctic-SincNet')
# signal = 'Arctic'

# En caso de usar una DNN
# sub = ['JG', 'RM'] # Dos pacientes (Arctic)
# path_audios_orig = '..\\SingleWordProductionDutch-main\\PMA\\Arctic'
# path_audios_sint = '.\\Resultados_audio_PMA_Arctic_DNN'
# output_folder_mfcc = '.\\Resultados_mfcc_PMA_Arctic_DNN\\'
# print('Resultados para Arctic-DNN')
# signal = 'Arctic'

## Para síntesis de los datos de EEG-M11 con SincNet ##
# sub = 'M11' 
# path_audios_orig = '..\\SingleWordProductionDutch-main\\HUVN\\'
# path_audios_sint = '.\\Resultados_audio_EEG_SincNet'
# output_folder_mfcc = '.\\Resultados_mfcc_EEG_SincNet'
# sesiones = ['ses01', 'ses02']


## Para síntesis de los datos de EEG-F09 con SincNet ##
# sub = 'F09' 
# path_audios_orig = '..\\SingleWordProductionDutch-main\\HUVN\\'
# path_audios_sint = '.\\Resultados_audio_EEG_SincNet'
# output_folder_mfcc = '.\\Resultados_mfcc_EEG_SincNet'
# sesiones = ['ses01', 'ses02']


## Para síntesis de los datos de EEG-M11 con DNN ##
# sub = 'M11' 
# path_audios_orig = '..\\SingleWordProductionDutch-main\\HUVN\\'
# path_audios_sint = '.\\Resultados_audio_EEG_DNN'
# output_folder_mfcc = '.\\Resultados_mfcc_EEG_DNN'
# sesiones = ['ses01', 'ses02']


## Para síntesis de los datos de EEG-F09 con DNN ##
sub = 'F09' 
path_audios_orig = '..\\SingleWordProductionDutch-main\\HUVN\\'
path_audios_sint = '.\\Resultados_audio_EEG_DNN'
output_folder_mfcc = '.\\Resultados_mfcc_EEG_DNN'
sesiones = ['ses01', 'ses02']


stoi_resultado = [] 
stoi_resultado_sujeto = []    

###### Para datos de PMA
# for i, sujeto in enumerate(sub):
#     for f in glob.iglob(os.path.join(path_audios_orig, sujeto, 'wav') + '\\*.wav'): # Se leen todos los audios sintetizados
#         elemento = os.path.basename(f) # Se obtiene el nombre de los audios para leer los originales correspondientes
        
#         rate, audio_original = waves.read(f) # Audio original
#         rate, audio_sintetizado = waves.read(os.path.join(path_audios_sint, elemento)) # Audio sintetizado correspondiente

#         length = min(len(audio_original), len(audio_sintetizado))
#         audio_original = audio_original[0:length]
#         audio_sintetizado = audio_sintetizado[0:length]

#         stoi_resultado.append(stoi(audio_original, audio_sintetizado, rate, extended=False))
#     stoi_resultado = np.array(stoi_resultado)
#     stoi_resultado_sujeto.append(np.mean(stoi_resultado))
#     stoi_resultado = []
# stoi_resultado_sujeto = np.array(stoi_resultado_sujeto)
    

# mcd = []
# mcd_sujeto = []
# for p_id, participant in enumerate(sub):
#     for f in glob.iglob(os.path.join(path_audios_orig, participant, 'sensor') + '\\*.npy'):
#         elemento = os.path.basename(f)
#         mfcc_original = np.load(os.path.join(path_audios_orig, participant, 'mfcc', elemento)) # Se lee cada señal PMA de cada palabra de cada uno de los participantes
#         mfcc_sint = np.load(os.path.join(output_folder_mfcc, elemento))
#         mcd.append(melCepstralDistance(mfcc_original, mfcc_sint))
#     mcd = np.array(mcd)
#     mcd_sujeto.append(np.mean(mcd))
#     mcd = []
# mcd_sujeto = np.array(mcd_sujeto)

# if signal == 'TiDigit':
#     print(f'Resultado MCD (sujeto LC): {mcd_sujeto[0]}')
#     print(f'Resultado MCD (sujeto TP): {mcd_sujeto[1]}')
#     print(f'Resultado MCD (media): {np.mean(mcd_sujeto)}')
#     print(f'Resultado STOI (sujeto LC): {stoi_resultado_sujeto[0]}')
#     print(f'Resultado STOI (sujeto TP): {stoi_resultado_sujeto[1]}')
#     print(f'Resultado STOI (media): {np.mean(stoi_resultado_sujeto)}')
# elif signal == 'Arctic':
#     print(f'Resultado MCD (sujeto JG): {mcd_sujeto[0]}')
#     print(f'Resultado MCD (sujeto RM): {mcd_sujeto[1]}')
#     print(f'Resultado MCD (media): {np.mean(mcd_sujeto)}')
#     print(f'Resultado STOI (sujeto JG): {stoi_resultado_sujeto[0]}')
#     print(f'Resultado STOI (sujeto RM): {stoi_resultado_sujeto[1]}')
#     print(f'Resultado STOI (media): {np.mean(stoi_resultado_sujeto)}')


###### Para datos de EEG
for k , sesion in enumerate(sesiones):
    for f in glob.iglob(os.path.join(path_audios_orig, sub, sesion, 'audio') + '\\*.wav'): # Se leen todos los audios sintetizados
        elemento = os.path.basename(f) # Se obtiene el nombre de los audios para leer los originales correspondientes
            
        rate, audio_original = waves.read(f) # Audio original
        rate, audio_sintetizado = waves.read(os.path.join(path_audios_sint, sub, sesion, elemento)) # Audio sintetizado correspondiente

        length = min(len(audio_original), len(audio_sintetizado))
        audio_original = audio_original[0:length]
        audio_sintetizado = audio_sintetizado[0:length]

        stoi_resultado.append(stoi(audio_original, audio_sintetizado, rate, extended=False))
    stoi_resultado = np.array(stoi_resultado)
    stoi_resultado_sujeto.append(np.mean(stoi_resultado))
    stoi_resultado = []
stoi_resultado_sujeto = np.array(stoi_resultado_sujeto)

mcd = []
mcd_sujeto = []
for k , sesion in enumerate(sesiones):
    for f in glob.iglob(os.path.join(output_folder_mfcc, sub, sesion) + '\\*.npy'):
        elemento = os.path.basename(f)
        mfcc_sint = np.load(f)
        mfcc_original = np.load(os.path.join(path_audios_orig, sub, sesion, 'mfcc', elemento))
        length = min(len(mfcc_sint[:,1]), len(mfcc_original[:,1]))
        mfcc_sint = mfcc_sint[0:length]
        mfcc_original = mfcc_original[0:length]
        mcd.append(melCepstralDistance(mfcc_original, mfcc_sint))
    mcd = np.array(mcd)
    mcd_sujeto.append(np.mean(mcd))
    mcd = []
mcd_sujeto = np.array(mcd_sujeto)

print(f'Resultado MCD (media): {np.mean(mcd_sujeto)}')
print(f'Resultado STOI (media): {np.mean(stoi_resultado_sujeto)}')
