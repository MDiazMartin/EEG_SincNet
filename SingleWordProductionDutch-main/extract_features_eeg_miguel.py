import os

import numpy as np 
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack

import glob as glob

#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)] # Funcion para calcular la señal analitica de una señal real

def extractHG(data, sr, windowLength, frameshift, sub):
    """
    Extracción de la envolvente en la banda de frecuencia utilizando la 
    transformada de Hilbert    
    
    Parámetros
    ----------
    data: array (samples, channels)
        Serie de datos temporales EEG
    sr: int
        Frecuencia de muestreo de los datos
    windowLength: float
        Duración de la ventana (en segundos) en la que se calculará el espectrograma
    frameshift: float
        Shift (en segundos) después de lo cual se extraerá la siguiente ventana
    Returns
    ----------
    feat: array (windows, channels)
        Matriz de características de la banda de frecuencia
    """
    # Eliminado de la tendencia lineal
    data = scipy.signal.detrend(data,axis=0)
    # Numero de ventanas
    numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    
    # Filtrado en la banda Gamma-Alta. Se diseña el filtro IIR
    if sub == 'M11':
        sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    elif sub == 'F09':
        sos = scipy.signal.iirfilter(4, [70/(sr/2),127/(sr/2)],btype='bandpass',output='sos')

    data = scipy.signal.sosfiltfilt(sos,data,axis=0) # Se aplica el filtro
    
    # Creación del espacio de características
    data = np.abs(hilbert3(data)) # Se calcula la señal analítica
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return feat


def reshapedata_2d(data):
    nchan, n_muestras = data.shape
    newdata = np.zeros((n_muestras, nchan))
    for i in range(0, n_muestras):
        newdata[i, :] = data[:, i].T
    return newdata


# Tamaño de ventana (en segundos)
windowLength = 0.05

# Desplazamiento entre ventanas (en segundos)
frameshift = 0.01

# Sujetos de la base de datos de eeg
sub = 'M11'
# sub = 'F09'

sesiones = ['ses01', 'ses02']

# Directorio de la base de datos
path_datos = '.\\HUVN' 

# Se crean los directorios de destino
for i in sesiones:
    os.makedirs(os.path.join(path_datos, sub, i, 'features'), exist_ok=True)

eeg_sr = 512 # Frecuencia de muestreo de la señal eeg. Para paciente M11 512 Hz, paciente F09 256 Hz.

# Se procesa cada archivo de audio de cada participante
for p_id, sesion in enumerate(sesiones):
    for f in glob.iglob(os.path.join(path_datos, sub, sesion, 'eeg') + '\\*.npy'):
        elemento = os.path.basename(f)
        
        eeg = np.load(f)
        eeg = eeg.squeeze(0)
        eeg = reshapedata_2d(eeg)

        # Atenuación del primer armónico del ruido de la línea
        sos = scipy.signal.iirfilter(4, [98/(eeg_sr/2),102/(eeg_sr/2)],btype='bandstop',output='sos')
        eeg = scipy.signal.sosfiltfilt(sos,eeg,axis=0) # Se aplica el filtro
            
        # Atenuación del segundo armónico del ruido de la línea
        if sub == 'M11':
            sos = scipy.signal.iirfilter(4, [148/(eeg_sr/2),152/(eeg_sr/2)],btype='bandstop',output='sos')
            eeg = scipy.signal.sosfiltfilt(sos,eeg,axis=0) # Se aplica el filtro

        # Extracción de las características HG
        feat = extractHG(eeg, eeg_sr, windowLength, frameshift, sub)

        np.save(os.path.join(path_datos, sub, sesion, 'features', elemento), feat)


    
    