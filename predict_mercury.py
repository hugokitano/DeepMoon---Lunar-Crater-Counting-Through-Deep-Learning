import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os
from PIL import Image
import pdb
from keras.models import load_model
import theano.ifelse
from utils import template_match_target as tmt
import input_data_gen as igen
import utils.processing as proc


def main():
   # pdb.set_trace()
    path = './input_data/'
    model = load_model('models/model_keras1.2.2.h5')

    #### TEST CRATERS
    test_imgs = h5py.File(path + '/Mercury_images.hdf5', 'r')

    images = test_imgs['input_images'][:100, :, :].astype('float32')
    test_data = {'imgs': [images[np.sum(images, axis = (1, 2)) > 0]]}
    #for img in test_data['imgs'][0]:
    #    print(np.sum(img), img[img > 0].shape)
    proc.preprocess(test_data)
    sd_input_images = test_data['imgs'][0]

    print(len(sd_input_images))
    images = [2, 10, 20, 44]
    plot_dir = "plots"

    for iwant in images:
        print("Predicting on image", iwant)
        pred = model.predict(sd_input_images[iwant:iwant + 1])
        extracted_rings = tmt.template_match_t(pred[0].copy(), minrad=2.)  # x coord, y coord, radius
        fig = plt.figure(figsize=[9, 9])
        [ax1, ax2, ax3] = fig.subplots(1, 3)
        ax1.imshow(sd_input_images[iwant].squeeze(), origin='upper', cmap='Greys_r', vmin=0, vmax=1.1)
        ax2.imshow(pred[0], origin='upper', cmap='Greys_r', vmin=0, vmax=1)
        ax3.imshow(sd_input_images[iwant].squeeze(), origin='upper', cmap="Greys_r")
        for x, y, r in extracted_rings:
            circle = plt.Circle((x, y), r, color='blue', fill=False, linewidth=2, alpha=0.5)
            ax3.add_artist(circle)
        ax1.set_title('Mercury DEM Image')
        ax2.set_title('CNN Predictions')
        ax3.set_title('Post-CNN Craters')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_location = os.path.join(current_dir, plot_dir + "/trained_model_results_Mercury" + str(iwant) + ".png")
        plt.savefig(plot_location)


if __name__ == '__main__':
    main()
