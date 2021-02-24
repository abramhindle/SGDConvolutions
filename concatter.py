#!/bin/env python3

import argparse
import numpy as np
import numpy
import scipy
import scipy.signal
import scipy.io.wavfile
from scipy.io import wavfile
from pathlib import Path
sr=48000
def parse_args():
    parser = argparse.ArgumentParser(description='concatenate sounds together')
    parser.add_argument('-i',nargs='+', help='input files')
    parser.add_argument('-a', default=0.05, help='attack in 0.0 to 1.0')
    parser.add_argument('-s', default=1.0, help='sustain')
    parser.add_argument('-r', default=0.1, help='release in 0.0 to 1.0')
    parser.add_argument('-out',default='output.wav', help='Output wav file')
    parser.add_argument('-sr',default=sr,help='assumed sample rate, try to use 48000')
    parser.add_argument('-n',default=True, help='normalize')
    args = parser.parse_args()
    return args

def normalize_wav(data):
    return data / max(abs(max(data)),abs(min(data)))

def concat_waves(inputs,outfilename,a=0.1,r=0.1,s=1.0,sr=48000,normalize=True):
    wavs = [scipy.io.wavfile.read(wav)[1] for wav in inputs]
    maxlen = sum([wav.shape[0] for wav in wavs])
    outwave = np.zeros(maxlen)
    window = np.ones(wavs[0].shape[0])
    # attack
    alen = int(float(a)*wavs[0].shape[0])
    # release
    rlen = int(float(r)*wavs[0].shape[0])
    sustain = float(s)
    window[0:alen] = np.linspace(0.0,sustain,alen)
    window[-rlen:] = np.linspace(sustain,0.0,rlen)
    index = 0
    for i, wav in enumerate(wavs):
        # assume same length
        outwave[index:index + wav.shape[0]] += (window * wav)
        index += (wav.shape[0] - rlen)
        print(index)
    index += 2*rlen
    # index points to the end
    if normalize:
        outwave = normalize_wav(outwave)
    wavfile.write(outfilename,int(sr),outwave[0:index])


if __name__ == "__main__":
    args = parse_args()
    concat_waves(args.i,args.out,
                 a=args.a, r=args.r, s=args.s, sr=args.sr,
                 normalize=args.n)
        
