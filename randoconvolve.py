#!/bin/env python
# use python3
import tensorflow as tf
import numpy as np
import numpy
import scipy
import scipy.signal
import scipy.io.wavfile
from scipy.io import wavfile
from pathlib import Path

# sr=11025
sr=48000

def check_or_make(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)    

def normalize_wav(data):
    return data / max(abs(max(data)),abs(min(data)))

outdir = "./sounds.out/"
outiirdir = "./iir.out/"
check_or_make(outdir)
check_or_make(outiirdir)

whatiwantfile = "whatiwant.wav"
whatiwantfile = "dijj001.11k.wav"
whatiwantfile = "quick-recorder.11k.wav"
whatiwantfile = "test123.wav"
inputwavfile = "test123.wav"
inputwavfile = "quick-recorder.11k.wav"
inputwavfile = "dijj001.wav"

_, whatiwant = wavfile.read(whatiwantfile)
whatiwant_sound = normalize_wav(whatiwant)
_, my_input = wavfile.read(inputwavfile)
my_input_sound = normalize_wav(my_input)

# 9000, 10000, 12000
MAXSIZE = 11025 # sr//5
MAXSIZE = 10240 # 1024*10
EPOCHS=100

def search_iteration(step=0,epochs=EPOCHS,maxsize=MAXSIZE,sr=sr,
                     my_input_sound=my_input_sound,
                     whatiwant_sound=whatiwant_sound):
    # same size
    n = min(maxsize,min(whatiwant_sound.shape[0],my_input_sound.shape[0]))
    print(f"n:{n}")
    if (step*n > min(whatiwant_sound.shape[0],my_input_sound.shape[0])):
        print("Too deep!")
        return False
    my_input =   my_input_sound[step*n:(step+1)*n]
    whatiwant = whatiwant_sound[step*n:(step+1)*n]
    # make tf constants
    expected_outputs_tensor_cons = tf.constant(whatiwant)
    expected_outputs_tensor = tf.reshape(expected_outputs_tensor_cons, [int(expected_outputs_tensor_cons.shape[0]), 1, 1],       name='expected_outputs')
    
    my_iir_tensor = tf.Variable(0.001*np.zeros((n,1,1)), name='kernel')
    # my input as a tensor
    inputs_tensor = tf.constant(my_input)
    my_inputs_r   = tf.reshape(inputs_tensor, [1, int(inputs_tensor.shape[0]), 1], name='inputs')
    custom_loss = lambda: \
        tf.math.reduce_sum(
            tf.math.square(
                tf.nn.conv1d(my_inputs_r, my_iir_tensor,stride=1,padding='SAME') - expected_outputs_tensor
            )/n
        )
    opt = tf.keras.optimizers.Adamax(learning_rate=0.01)
    my_iir_tensor = tf.Variable(0.001*np.zeros((n,1,1)), name='kernel')
    iirs = []
    for i in range(epochs):
        step_count = opt.minimize(custom_loss, [my_iir_tensor]).numpy()
        current_loss = custom_loss().numpy()
        print(f"Step: {step} Epoch: {step_count} Loss:{current_loss}")
        iirs.append(my_iir_tensor.numpy())        
    def iir_convolve(iir,my_inputs_r,n):
        '''iir.shape is (n,1,1)'''
        my_iir_tensor = tf.constant(iir.reshape(n,1,1))
        res = tf.nn.conv1d(my_inputs_r, my_iir_tensor, stride=1, padding='SAME')
        return res.numpy()
    audios = [normalize_wav(iir_convolve(iir,my_inputs_r,n).reshape(n)) for iir in iirs]
    for i,audio in enumerate(audios):
        wavfile.write(f"{outdir}/snippet-adamax.{step:05}.{i:05}.wav",sr,audio)
    normalized_audios = [normalize_wav(wav) for wav in audios]
    audio = np.concatenate(audios)
    normalized_audio = np.concatenate(normalized_audios)
    wavfile.write(f"{outdir}/plain-adamax.{step:05}.wav",sr,audio)
    wavfile.write(f"{outdir}/norm-adamax.{step:05}.wav",sr,normalized_audio)
    for i,iir in enumerate(iirs):
        iira = iir.reshape(n)
        wavfile.write(f"{outiirdir}/iir-adamax.{step:05}.{i:05}.wav",sr,iir)
    return True

current_step = 0
while(search_iteration(step=current_step)):
    print(f"Step {current_step}")
    current_step += 1
