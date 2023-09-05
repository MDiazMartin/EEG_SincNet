##### Script para los datos de EEG haciendo uso de una red neuronal basada en SincNet #####

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import random
import os
from math import floor
from math import ceil
import glob as glob

from Modelo_SincNet import *

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


def reshapedata(data):
    timestep, nchan, ntrial = data.shape
    newdata = np.zeros((ntrial, nchan, timestep))
    for i in range(0, ntrial):
        newdata[i, :, :] = data[:, :, i].T
    return newdata

def reshapedata_2d(data):
    nchan, n_muestras = data.shape
    newdata = np.zeros((n_muestras, nchan))
    for i in range(0, n_muestras):
        newdata[i, :] = data[:, i].T
    return newdata

def resetea_pesos(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # Resetea los pesos y sesgos de las capas convolucionales y lineales
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            # Resetea los pesos y sesgos de las capas de Batch Normalization
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    
# Dummy training datasets
wlen_mfcc = 25
N_channels = 90 # 77 para sujeto M11, 90 para sujeto F09

# Dimensiones de entrada y salida de la red
input_dim = N_channels
output_dim = wlen_mfcc

# Ratio de aprendizaje
Learn_R = 5e-4

# Creación del modelo
model = SincDriftBoundAttChoice_full(dropout=0.1)

# Parámetros de cada capa y totales
for nombre_capa, parametros in model.named_parameters():
    print(f"{nombre_capa}: {torch.numel(parametros)}")
total_parametros = sum(parametros_capa.numel() for parametros_capa in model.parameters())
print(f"Total de parámetros en la CNN: {total_parametros}")

# Definir la función de pérdida y el optimizador
loss = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=Learn_R, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=Learn_R, weight_decay=0)

# sub = 'M11'
sub = 'F09'

sesiones = ['ses01', 'ses02']

# Directorios
path_datos = '..\\SingleWordProductionDutch-main\\HUVN\\'

# output_folder_mfcc = '.\\Resultados_mfcc_EEG_SincNet\\M11'
# output_folder_audio = '.\\Resultados_audio_EEG_SincNet\\M11'

output_folder_mfcc = '.\\Resultados_mfcc_EEG_SincNet\\F09'
output_folder_audio = '.\\Resultados_audio_EEG_SincNet\\F09'

# Se crean los directorios
for p_id, sesion in enumerate(sesiones):
    os.makedirs(os.path.join(output_folder_mfcc, sesion), exist_ok=True)
    os.makedirs(os.path.join(output_folder_audio, sesion), exist_ok=True)

# fs_eeg = 512 # Frecuencia de muestreo de la señal EEG para el paciente M11
fs_eeg = 256 # Frecuencia de muestreo de la señal EEG para el paciente F09
wlen = 50e-3 # tamaño (en ms) de las ventanas
frameshift = 10e-3 # Tamaño del paso entre tramas (en ms).

num_muestras_eeg = ceil(fs_eeg * wlen)
num_muestras_eeg_desplazamiento = floor(fs_eeg * frameshift) # Cada num_tramas_eeg se calculan los MFCC

z=0

# Se procesa cada archivo de cada sesion del participante
for p_id, sesion in enumerate(sesiones):
    for f in glob.iglob(os.path.join(path_datos, sub, sesion, 'audio') + '\\*.wav'):

        if 'eeg' in locals(): # Se comprueba que existe la variable para apilar los datos

            elemento = os.path.basename(f).split('.')[0]
            elemento_mfcc = elemento + '.npy'
            elemento_eeg = elemento.split('_')[0] + '_' + elemento.split('_')[1] + '_S.npy'
            eeg = np.load(os.path.join(path_datos, sub, sesion, 'eeg', elemento_eeg)) # Se lee cada señal EEG de cada palabra
            eeg = eeg.squeeze(0)
            eeg = reshapedata_2d(eeg)
            
            MFCC_aux = np.load(os.path.join(path_datos, sub, sesion, 'mfcc', elemento_mfcc))

            # Se truncan para asegurar que tienen la misma longitud
            eeg = eeg[0:ceil(num_muestras_eeg_desplazamiento*145) + num_muestras_eeg,:]
            MFCC_aux = MFCC_aux[0:145, :]

            for i in range(int((np.shape(eeg)[0]-num_muestras_eeg)/num_muestras_eeg_desplazamiento)):
                eeg_aux2 = eeg[i * num_muestras_eeg_desplazamiento : i * num_muestras_eeg_desplazamiento + num_muestras_eeg]
                eeg_aux2 = np.expand_dims(eeg_aux2, axis=2)
                eeg_envent = np.concatenate((eeg_envent, eeg_aux2), axis=2)

            MFCC = np.concatenate((MFCC, MFCC_aux))

        else:
            elemento = os.path.basename(f).split('.')[0]
            elemento_mfcc = elemento + '.npy'
            elemento_eeg = elemento.split('_')[0] + '_' + elemento.split('_')[1] + '_S.npy'
            eeg = np.load(os.path.join(path_datos, sub, sesion, 'eeg', elemento_eeg)) # Se lee cada señal EEG de cada palabra
            eeg = eeg.squeeze(0)
            eeg = reshapedata_2d(eeg)

            MFCC = np.load(os.path.join(path_datos, sub, sesion, 'mfcc', elemento_mfcc))
                
            # Se truncan para asegurar que tienen la misma longitud
            eeg = eeg[0:ceil(num_muestras_eeg_desplazamiento*145) + num_muestras_eeg,:]
            MFCC = MFCC[0:145,:]

            for i in range(int((np.shape(eeg)[0]-num_muestras_eeg)/num_muestras_eeg_desplazamiento)):
                if z==0:
                    eeg_envent = eeg[0:num_muestras_eeg, :]
                    eeg_envent = np.expand_dims(eeg_envent, axis=2)
                    z=1
                else:
                    eeg_aux2 = eeg[i * num_muestras_eeg_desplazamiento : i * num_muestras_eeg_desplazamiento + num_muestras_eeg]
                    eeg_aux2 = np.expand_dims(eeg_aux2, axis=2)
                    eeg_envent = np.concatenate((eeg_envent, eeg_aux2), axis=2)

eeg = eeg_envent
eeg = reshapedata(eeg)

scaler_salida = StandardScaler()
scaler_salida.fit(MFCC)
MFCC = scaler_salida.transform(MFCC)

# Tamaño del batch y numero de batches
batch_size = 64
num_batches = eeg.shape[0]/batch_size

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=False) # Para la división del dataset en train, test

torch.manual_seed(123123)
bloques_eeg = np.shape(eeg)[0]
# Se entrena la red y se evalua haciendo uso de la validacion cruzada
i=0
for (train_index, test_index) in kf.split(range(bloques_eeg)): 
    random.shuffle(train_index) # Aleatorizar los índices para establecer conjunto de entrenamiento y validacion

    eval_index = train_index[0:floor(len(train_index)/10)] # indices de evaluacion
    train_index = train_index[ceil(len(train_index)/10):len(train_index)] # indices de entrenamiento

    # Los datos se convierten en tensores de Pytorch
    eeg_train = torch.tensor(eeg[train_index,:,:], dtype=torch.float32)
    MFCC_train = torch.tensor(MFCC[train_index], dtype=torch.float32)
    MFCC_train = MFCC_train.unsqueeze(2) # Se aumenta la dimension para poder crear el dataset

    eeg_eval = torch.tensor(eeg[eval_index,:,:], dtype=torch.float32)
    MFCC_eval = torch.tensor(MFCC[eval_index], dtype=torch.float32)
    MFCC_eval = MFCC_eval.unsqueeze(2)

    eeg_test = torch.tensor(eeg[test_index,:,:], dtype=torch.float32)
    MFCC_test = torch.tensor(MFCC[test_index], dtype=torch.float32)
    MFCC_test = MFCC_test.unsqueeze(2)

    # Se construye el dataset y el dataloader para facilitar la carga de datos en el entrenamiento y test
    train_set = TensorDataset(eeg_train, MFCC_train)
    validation_set = TensorDataset(eeg_eval, MFCC_eval)
    # test_set = TensorDataset(X_feat_test, Y_MFCC_test)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Errores medios en cada época para conjuntos de entrenamiento y evaluación
    train_loss_acumulada = []
    validation_loss_acumulada = []
    
    # Parametros para early stopping
    N_epocas = 100
    paciencia = 8
    contador_early = 0
    mejores_pesos = model.state_dict()
    mejor_metrica = float('inf') # Para acumular 
    mejor_epoca = 0

    resetea_pesos(model) # Se resetean los pesos de la red para entrenar la red de nuevo con los nuevos datos del pliegue

    for epoch in range(N_epocas):
        print(f'Época {epoch+1}: cargando...')
        model.train() # Se pone el modelo en modo de entrenamiento

        # Entrenamiento
        train_loss_acumulada_aux = 0.0
        for z, (inputs_real, outputs_real) in enumerate(train_loader): # Para cargar cada batch
            
            outputs_real = outputs_real.squeeze(2)

            outputs = model(inputs_real) # Se obtienen las predicciones

            train_loss = loss(outputs, outputs_real) # Se obtiene el error

            optimizer.zero_grad() # Se reinician los gradientes fijandoles valor 0

            train_loss.backward() # Se retropropaga el error
            optimizer.step() # Se optimizan los pesos

            train_loss_acumulada_aux = train_loss_acumulada_aux + train_loss.item()

        # Error medio de los lotes de entrenamiento
        train_loss_acumulada_aux = train_loss_acumulada_aux / len(train_loader) # Error medio de los lotes
        train_loss_acumulada.append(train_loss_acumulada_aux) # Se acumula el error de la época actual

        # Validacion
        model.eval() # Se pone el modelo en modo de evaluación
        validation_loss_acumulada_aux = 0.0
        with torch.no_grad():
            for z, (inputs_real, outputs_real) in enumerate(validation_loader): # Para cargar cada batch
                outputs_real = outputs_real.squeeze(2)
                outputs = model(inputs_real)
                validation_loss = loss(outputs, outputs_real) # Se obtiene el error del conjunto de validación
                validation_loss_acumulada_aux = validation_loss_acumulada_aux + validation_loss.item() 

        # Error medio de los lotes de validacion
        validation_loss_acumulada_aux = validation_loss_acumulada_aux / len(validation_loader) # Error medio de los lotes
        validation_loss_acumulada.append(validation_loss_acumulada_aux) # Se acumula el error de la época actual

        # Comprobar si se ha mejorado la métrica de validación
        if validation_loss_acumulada_aux < mejor_metrica:
            mejores_pesos = model.state_dict()
            mejor_metrica = validation_loss_acumulada_aux
            mejor_epoca = epoch
            contador_early = 0
        else:
            contador_early = contador_early + 1
        
        # Se puede mostrar el progreso de las pérdidas
        print(f'Época {epoch + 1}/{N_epocas}, Pérdida de entrenamiento: {train_loss_acumulada_aux:.4f}, Pérdida de validación: {validation_loss_acumulada_aux:.4f}')

        # Verificar si se ha alcanzado el criterio de early stopping
        if contador_early >= paciencia:
            print(f'Early stopping en la época {epoch+1}')
            break # Para salir del bucle que recorre las épocas

    # Se cargan los pesos del modelo con mejores resultados
    model.load_state_dict(mejores_pesos) 

    # Guardar el modelo entrenado
    # torch.save(model, '.\\Modelo guardado\\mymodel_DNN.pt')

    model.eval() # Se pone el modelo en modo de evaluación
    
    # Obtención de los MFCC de cada pliegue en la validación cruzada
    with torch.no_grad():
        outputs = model(eeg_test)
        if i == 0:
            MFCC_pred = np.array(outputs, dtype='double')
            MFCC_pred = scaler_salida.inverse_transform(MFCC_pred)
            i=i+1
        else:
            MFCC_pred_aux = np.array(outputs, dtype='double')
            MFCC_pred_aux = scaler_salida.inverse_transform(MFCC_pred_aux)
            MFCC_pred = np.concatenate((MFCC_pred, MFCC_pred_aux))
            i=i+1

    print(f'Etapa {i} de la validación cruzada terminada \n')

# Teniendo en cuenta la longitud de cada secuencia, se almacenan los MFCC obtenidos de cada palabra en el directorio de destino
iter = 0
for p_id, sesion in enumerate(sesiones):
    for f in glob.iglob(os.path.join(path_datos, sub, sesion, 'audio') + '\\*.wav'):
        elemento = os.path.basename(f).split('.')[0]
        elemento = elemento + '.npy'
        np.save(os.path.join(output_folder_mfcc, sesion, elemento), MFCC_pred[145*iter:145*(iter+1),:])
        iter = iter +1

# Se calcula la métrica MCD (Mel Cepstral Distortion)
mcd = []
for p_id, sesion in enumerate(sesiones):
    for f in glob.iglob(os.path.join(path_datos, sub, sesion, 'audio') + '\\*.wav'):
        elemento = os.path.basename(f).split('.')[0]
        elemento = elemento + '.npy'
        mfcc_original = np.load(os.path.join(path_datos, sub, sesion, 'mfcc', elemento)) # Se leen los MFCC de cada palabra sintetizada
        mfcc_sint = np.load(os.path.join(output_folder_mfcc, sesion, elemento))
        mcd.append(melCepstralDistance(mfcc_original, mfcc_sint))

mcd = np.array(mcd)

##### Representación del MCD para cada pliegue #####
mcd_pliegues = []
for g in range(n_splits):
    mcd_pliegues.append(np.mean(mcd[int(g*len(mcd)/n_splits):int((g+1)*len(mcd)/n_splits)]))
mcd_pliegues = np.array(mcd_pliegues)

plt.figure()
pliegues = [f"Pliegue {i+1}" for i in range(n_splits)]

plt.bar(pliegues, mcd_pliegues, align="center")
plt.xticks(rotation=45) # Rotacion de las etiquetas del eje x
plt.ylabel("Valor de la métrica MCD")

plt.title("Valor medio de MCD para cada pliegue")
plt.gca().yaxis.grid(True, linestyle='--')
plt.show()

mcd_mean = np.mean(mcd) # Se calcula la media de todos los MCD de todos los MFCC generados con respecto a sus correspondientes audios originales
np.save('.\\MCD_EEG_SincNet', mcd_mean) # Se guarda el resultado para representarlo gráficamente después

print(f'El MCD promedio es: {mcd_mean}')

##### Representación del error para cada época en el último pliegue #####
plt.figure()

plt.xlabel('Época')
plt.ylabel('Error MSE')
plt.title("Evolucion del error de validación y de entrenamiento")
plt.plot(train_loss_acumulada, label='Error de entrenamiento')
plt.plot(validation_loss_acumulada, label='Error de validación')
plt.legend()
plt.show()