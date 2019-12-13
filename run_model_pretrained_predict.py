#!/usr/bin/env python
"""Run Convolutional Neural Network Training

Execute the training of a (UNET) Convolutional Neural Network on
images of the Moon and binary ring targets.
"""

import model_train as mt

# Model Parameters
MP = {}

# Directory of train/dev/test image and crater hdf5 files.
MP['dir'] = 'mars_data/'

# Image width/height, assuming square images.
MP['dim'] = 256

# Min and max radius of craters to detect
MP['minrad'] = 10
MP['maxrad'] = 50

# Number of train/valid/test samples, needs to be a multiple of batch size.
MP['n_test'] = 5000

# path to load model from.
MP['model_path'] = 'models/model_30k.h5'


# Iterating over parameters example.
#    MP['N_runs'] = 2
#    MP['lambda']=[1e-4,1e-4]

if __name__ == '__main__':
    mt.predict_using_pretrained_model(MP)

