'''Este script contiene el modelo de red neuronal utilizado para sintetizar los coeficientes
MFCC correspondientes a una ventana temporal de datos de EEG o PMA'''

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math
from layers_sinc_spatial import *

# torch.manual_seed(2022)
# np.random.seed(2022)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class SincDriftBoundAttChoice_full(nn.Module):

    # Parámetros para PMA-TiDigit
    num_filters = 40 # Numero de filtros utilizados para extraer características. Cuantos mas, mayor complejidad.
    filter_length = 7 # Longitud de los filtros en muestras
    t_length = 10 # Igual al numero de muestras que se van a procesar
    pool_window_ms = 40  # Tamaño de la ventana de pooling.
    stride_window_ms = 10 # Tamaño del paso de la ventana de pooling.
    num_chan = 9 # Numero de canales. 
    spatialConvDepth = 1
    sr = 100 
    cutoff = 10  # can set it to nyquist
    pool_window = int(np.rint(((100 - filter_length +1) * pool_window_ms)/ (1000)))
    stride_window =  int(np.rint(((100 - filter_length +1) * stride_window_ms)/ (1000)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((10 - filter_length +1) - pool_window)/stride_window +1))


    # Parámetros para EEG
    # num_filters = 40 # Numero de filtros utilizados para extraer características. Cuantos mas, mayor complejidad.
    # filter_length = 19 # Longitud de los filtros en muestras: 19 para sujeto M11, 11 para sujeto F09
    # t_length = 26 # 13 para sujeto F09, 26 para sujeto M11
    # pool_window_ms = 20  # Tamaño de la ventana de pooling.
    # stride_window_ms = 10  # Tamaño del paso de la ventana de pooling. 
    # num_chan = 77 # 90 para sujeto F09, 77 para sujeto M11
    # spatialConvDepth = 1
    # sr = 512 # 512 para paciente M11, 256 para F09
    # cutoff = 55 # can set it to nyquist
    # pool_window = int(np.rint(((200- filter_length +1) * pool_window_ms)/ (1000)))
    # stride_window =  int(np.rint(((200- filter_length +1) * stride_window_ms)/ (1000)))
    # if pool_window % 2 == 0 :
    #     pool_window -= 1
    # if stride_window % 2 == 0 :
    #     stride_window -= 1
    # output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    
    def __init__(self, dropout):
        super(SincDriftBoundAttChoice_full, self).__init__()

        self.b0 = nn.BatchNorm2d(1, momentum=0.99) 

        self.sinc_cnn2d = sinc_conv(self.num_filters, self.filter_length, self.sr, cutoff=self.cutoff) 

        self.b =  nn.BatchNorm2d(self.num_filters, momentum=0.99)

        self.separable_conv = SeparableConv2d(self.num_filters, self.num_filters, depth = self.spatialConvDepth, kernel_size= (self.num_chan,1))

        self.b2 = nn.BatchNorm2d(self.num_filters*self.spatialConvDepth, momentum=0.99)

        self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))

        self.dropout1 = torch.nn.Dropout(p=dropout)

        self.fc_mfcc = torch.nn.Linear(self.num_filters*self.spatialConvDepth*self.output_size,25)

    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad


    def forward(self, x):
        # print("Input: ", x.shape)
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan, self.t_length)
        x = self.b0(x)
        # print("BatchNorm1: ", x.shape)
        x = torch.squeeze(x)
        if batch_n > 1:
            x = torch.squeeze(x)
        else:
            x = x.view(batch_n,self.num_chan, self.t_length)
        x0 = self.sinc_cnn2d(x)
        # print("sinc_conv: ", x0.shape)

        score0 = self.b(x0)
        # print("BatchNorm2: ", score0.shape)

        score0 = self.separable_conv(score0)
        # print("conv_2d: ", score0.shape)

        score = self.b2(score0)
        # print("BatchNorm3: ", score.shape)

        score = F.relu(score)

        score = self.pool1(score)
        # print("pool: ", score.shape)

        score = self.dropout1(score) 
        # print("dropout: ", score.shape)

        score2 = score.view(batch_n, -1)

        mfcc = self.fc_mfcc(score2)
        # print("DNN: ", mfcc.shape)

        return mfcc

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp







    