#!/usr/bin/env python3

import optparse
import numpy as np
import pyworld as pw
import sys
import os.path
from glob import iglob
import soundfile as sf
# from nnmnkwii.preprocessing.f0 import interp1d
from librosa import resample
from scipy import interpolate

_DEFAULT_FRAME_PERIOD = 10
#_DEFAULT_FRAME_PERIOD = 5
_DEFAULT_MFCC_ORDER = 25
_TARGET_FS = 16000
_F0_FLOOR = 80
_F0_CEIL = 650

path_param_sint = '..\\SingleWordProductionDutch-main\\PMA\\TiDigit'
sub = ['LC', 'TP']

def interp_f0(f0, kind='slinear'):
    idx_voiced = np.where(f0 > 0)[0]
    idx_unvoiced = np.where(f0 == 0)[0]
    f = interpolate.interp1d(idx_voiced, f0[idx_voiced], kind='slinear', bounds_error=False, fill_value='extrapolate')
    f0_interp = f0.copy()
    f0_interp[idx_unvoiced] = f(idx_unvoiced)
    return f0_interp
    

def print_usage():
    print("Usage: {} [OPTIONS] src_dir src_ext trg_dir\n".format(sys.argv[0]))
    print("")
    print("- src_dir: directory with the source files (either wav or npy files)")
    print("- src_ext: extension of the source files (either wav or npy). Depending of the type of input files, different processing is performed:")
    print("\t.wav files: the program uses the Word vocoder to extract a set of parameters and stores this parameter in the trg_dir as npy files")
    print("\t.npy files: the program uses the Word vocoder to synthesise speech from the provided .npy files")
    print("- trg_dir: directory where the target files are saved (either wav or npy files)")
    print("")
    print("OPTIONS are:")
    print("\t--frame_period: frame period used for computing the speech features [Default={}]".format(_DEFAULT_FRAME_PERIOD))
    print("\t--num_mfcc: number of MFCCs used to represent the spectral envelope [Default={}]\n".format(_DEFAULT_MFCC_ORDER))


def parse_args():
    # Parse the program args
    p = optparse.OptionParser()
    p.add_option("--frame_period", type="int", default=_DEFAULT_FRAME_PERIOD)
    p.add_option("--num_mfcc", type="int", default=_DEFAULT_MFCC_ORDER)
    opt, args = p.parse_args()
    return opt, args[0], args[1], args[2]


def world(x, fs=16000, frame_period=10):
    '''
    Función para codificar una señal de audio x muestreada a fs Hz con el vocoder World (https://github.com/mmorise/World)

    Parameters:
    x: señal de voz.
    fs: frecuencia de muestreo.
    frame_period: indica cada cuánto se calculan las tramas (en ms). Valores recomendados: 5 ó 10 ms.

    Returns:
    f0: Vector con los valores de F0 de la señal (Tam: num_frames).
    sp: Matrix 2d con el espectro suavizado de la señal (Tam: num_frames * 513).
    ap: Matriz de aperiodicidades (Tam: num_frames * 513).
    '''
    waveform = x.astype(dtype=np.double)
    _f0, t = pw.harvest(waveform, fs, f0_floor=_F0_FLOOR, f0_ceil=_F0_CEIL, frame_period=frame_period)    # raw pitch extractor
    f0 = pw.stonemask(waveform, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(waveform, f0, t, fs, f0_floor=_F0_FLOOR)  # extract smoothed spectrogram
    ap = pw.d4c(waveform, f0, t, fs)         # extract aperiodicity    
    return f0, sp, ap


def code_world_params(f0, sp, ap, fs=16000, num_mfcc=25):
    '''
    Comprime los parámetros extraídos por World.

    Parameters:
    f0: vector con los valores de F0, incluidos las etiquetas Sonoro/Sordo.
    sp: Matrix 2d con el espectro suavizado de la señal.
    ap: Matriz de aperiodicidades.
    fs: frecuencia de muestreo.
    num_mfcc: Número de MFCC que se calculan por trama a partir de la matriz sp.


    Returns:
    lf0: Vector con los valores de F0 interpolados en escala logarítmica.
    vuv: Vector con las decisiones sonoro/sordo.
    mfcc: Matriz de MFCCs
    bap: Matriz con las band-aperiodicities.
    '''
    mfcc = pw.code_spectral_envelope(sp, fs, num_mfcc)
    bap = pw.code_aperiodicity(ap, fs)
    vuv = (f0 > 0).astype(np.float32)
    # Interpolate the F0 for the unvoiced segments and apply a log-compression
    lf0 = np.log(interp_f0(f0, kind='slinear') + 1e-6)
    return lf0, vuv, mfcc, bap


def decode_world_params(lf0, vuv, mfcc, bap, fs=16000):
    '''
    Decodifica (descomprime) los parámetros comprimidos por la función code_world_params para
    ser usados para sintetizar voz.

    Parameters:
    lf0: Vector con los valores de F0 interpolados en escala logarítmica.
    vuv: Vector con las decisiones sonoro/sordo.
    mfcc: Matriz de MFCCs
    bap: Matriz con las band-aperiodicities.
    fs: frecuencia de muestreo.

    Returns:
    f0: vector con los valores de F0, incluidos las etiquetas Sonoro/Sordo.
    sp: Matrix 2d con el espectro suavizado de la señal.
    ap: Matriz de aperiodicidades.
    '''
    fft_size = pw.get_cheaptrick_fft_size(fs)
    sp = pw.decode_spectral_envelope(mfcc.copy(order='C'), fs=fs, fft_size=fft_size)
    ap = pw.decode_aperiodicity(bap.copy(order='C'), fs=fs, fft_size=fft_size)
    f0 = np.exp(lf0) * vuv
    return f0, sp, ap


def synthesise(f0, sp, ap, fs=16000, frame_period=10):
    '''
    Resintetiza una señal de voz a partir de los parámetros extraídos por World.

    Parameters:
    f0: vector con los valores de F0, incluidos las etiquetas Sonoro/Sordo.
    sp: Matrix 2d con el espectro suavizado de la señal.
    ap: Matriz de aperiodicidades.
    fs: frecuencia de muestreo.
    frame_period: indica cada cuánto se calculan las tramas (en ms). Valores recomendados: 5 ó 10 ms.

    Returns:
    x: señal en el tiempo.
    '''

    minimo = min([np.shape(f0)[0], np.shape(sp)[0], np.shape(ap)[0]])

    f0 = f0[0:minimo]
    sp = sp[0:minimo,:]
    ap = ap[0:minimo,:]
    x = pw.synthesize(f0.flatten(), sp, ap, fs=fs, frame_period=frame_period)
    return x


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)

    # Parse the command line arguments
    (opt, src_dir, src_ext, trg_dir) = parse_args()

    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)

    ext = src_ext.lower().lstrip('.')
    if ext == 'wav':
        for filepath in iglob(os.path.join(src_dir, "*.wav")):
            print("Processing file {}...".format(os.path.basename(filepath)))
            x, fs = sf.read(filepath)
            if fs != _TARGET_FS:
                x = resample(x, fs, _TARGET_FS)
                fs = _TARGET_FS
            f0, sp, ap = world(x, fs, frame_period=opt.frame_period)
            lf0, vuv, mfcc, bap = code_world_params(f0, sp, ap, fs, opt.num_mfcc)
            # Save the features
            filename = os.path.splitext(os.path.basename(filepath))[0]
            # np.save(os.path.join(trg_dir, filename + ".npy"), np.hstack([mfcc, bap, lf0[:, np.newaxis], vuv[:, np.newaxis]]))
            np.save(os.path.join(trg_dir, filename + ".npy"), mfcc)

    elif ext == 'npy':
        for filepath in iglob(os.path.join(src_dir, "*.npy")):
            print("Processing file {}...".format(os.path.basename(filepath)))
            x = np.load(filepath)
            # mfcc = x
            # nframes = x.shape[0]
            # bap = np.zeros(shape=(nframes, 1), dtype=float)
            # lf0 = np.zeros_like(bap)
            # vuv = np.zeros_like(lf0)
            # Unpack the parameters
            mfcc = x
            if os.path.exists(os.path.join(path_param_sint, sub[0], "mfcc", os.path.basename(filepath))):
                bap = np.load(os.path.join(path_param_sint, sub[0], "mfcc", os.path.basename(filepath).split('.')[0] + "_bap.npy"))
                lf0 = np.load(os.path.join(path_param_sint, sub[0], "mfcc", os.path.basename(filepath).split('.')[0] + "_lf0.npy"))
                vuv = np.load(os.path.join(path_param_sint, sub[0], "mfcc", os.path.basename(filepath).split('.')[0] + "_vuv.npy"))
            elif os.path.exists(os.path.join(path_param_sint, sub[1], "mfcc", os.path.basename(filepath))):
                bap = np.load(os.path.join(path_param_sint, sub[1], "mfcc", os.path.basename(filepath).split('.')[0] + "_bap.npy"))
                lf0 = np.load(os.path.join(path_param_sint, sub[1], "mfcc", os.path.basename(filepath).split('.')[0] + "_lf0.npy"))
                vuv = np.load(os.path.join(path_param_sint, sub[1], "mfcc", os.path.basename(filepath).split('.')[0] + "_vuv.npy"))
            else:
                nframes = x.shape[0]
                bap = np.zeros(shape=(nframes, 1), dtype=float)
                lf0 = np.zeros_like(bap)
                vuv = np.zeros_like(lf0)
            # Decode them
            f0, sp, ap = decode_world_params(lf0, vuv, mfcc, bap, fs=_TARGET_FS)
            waveform = synthesise(f0, sp, ap, fs=_TARGET_FS, frame_period=opt.frame_period)
            # Save the waveform
            filename = os.path.splitext(os.path.basename(filepath))[0]
            sf.write(os.path.join(trg_dir, filename + ".wav"), waveform, _TARGET_FS)
    else:
        raise RuntimeError('Unsuported file type: {}'.format(ext))
