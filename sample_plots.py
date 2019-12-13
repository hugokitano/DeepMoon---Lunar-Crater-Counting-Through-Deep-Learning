import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os
from PIL import Image
import pdb
from keras.models import load_model
from utils import template_match_target as tmt
import input_data_gen as igen
import utils.processing as proc


def main():
    # pdb.set_trace()
    zenodo_path = './mars_data/'
    model = load_model('models/model_30k.h5')

       #### TEST CRATERS 
    test_imgs = h5py.File(zenodo_path + '/mars_images_5k.hdf5', 'r')

    test_data = {'imgs': [test_imgs['input_images'][...].astype('float32'),
                        test_imgs['target_masks'][...].astype('float32')]}
    proc.preprocess(test_data)
    sd_input_images = test_data['imgs'][0]
    sd_target_masks = test_data['imgs'][1]

    images = [25,36]
    plot_dir = "plots"

    for iwant in images:
        pred = model.predict(sd_input_images[iwant:iwant + 1])
        extracted_rings = tmt.template_match_t(pred[0].copy(), minrad=2.) # x coord, y coord, radius

        fig = plt.figure(figsize=[16, 16])
        plt.rcParams["font.size"] = 20
        [[ax1, ax4], [ax2, ax3]] = fig.subplots(2, 2)
        ax1.imshow(sd_input_images[iwant].squeeze(), origin='upper', cmap='Greys_r', vmin=0, vmax=1.1)
        ax2.imshow(sd_target_masks[iwant].squeeze(), origin='upper', cmap='Greys_r')
        ax3.imshow(pred[0], origin='upper', cmap='Greys_r', vmin=0, vmax=1)
        ax4.imshow(sd_input_images[iwant].squeeze(), origin='upper', cmap="Greys_r")
        for x, y, r in extracted_rings:
            circle = plt.Circle((x, y), r, color='blue', fill=False, linewidth=2, alpha=0.5)
            ax4.add_artist(circle)
        ax1.set_title('Mars DEM Image')
        ax2.set_title('Ground-Truth Target Mask')
        ax3.set_title('CNN Predictions')
        ax4.set_title('Post-CNN Craters')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_location = os.path.join(current_dir, plot_dir + "/trained_model_results" + str(iwant) + ".png")
        plt.savefig(plot_location)

    

    
    

if __name__ == '__main__':
    main()