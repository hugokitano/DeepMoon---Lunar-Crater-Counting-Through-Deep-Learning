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
    zenodo_path = './data/'
    deepmoon_path = os.path.dirname(os.getcwd())
    train_imgs = h5py.File(zenodo_path + 'train_images.hdf5', 'r')
    fig = plt.figure(figsize=[12, 6])
    [ax1, ax2] = fig.subplots(1,2)
    ax1.imshow(train_imgs['input_images'][3][...], origin='upper', cmap='Greys_r', vmin=120, vmax=200)
    ax2.imshow(train_imgs['target_masks'][3][...], origin='upper', cmap='Greys_r')
    plt.show()

    ctrs = pd.HDFStore(zenodo_path + 'train_craters.hdf5', 'r')

    pdb.set_trace()

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(zenodo_path + "/LunarLROLrocKaguya_118mperpix.png").convert("L")
    # Read and combine the LROC and Head datasets (stored under ../catalogues)
    craters = igen.ReadLROCHeadCombinedCraterCSV(filelroc="catalogues/LROCCraters.csv",
                                             filehead="catalogues/HeadCraters.csv")

    # Generate 100 image/target sets, and corresponding crater dataframes.  np.random.seed is set for consistency.
    igen.GenDataset(img, craters, zenodo_path + '/test_zenodo', amt=25, seed=1337)

    gen_imgs = h5py.File(zenodo_path + '/test_zenodo_images.hdf5', 'r')
    sample_data = {'imgs': [gen_imgs['input_images'][...].astype('float32'),
                            gen_imgs['target_masks'][...].astype('float32')]}
    proc.preprocess(sample_data) #now, input images is shape 25 x 256 x 256 x 1, target_masks is shape 25 x 256 x 256
    sd_input_images = sample_data['imgs'][0] #25 x 256 x 256 x 1 
    sd_target_masks = sample_data['imgs'][1] #25 x 256 x 256

    # Plot the data for fun.
    fig = plt.figure(figsize=[12, 6])
    [ax1, ax2] = fig.subplots(1,2)
    ax1.imshow(sd_input_images[5].squeeze(), origin='upper', cmap='Greys_r', vmin=0, vmax=1.1)
    ax2.imshow(sd_target_masks[5].squeeze(), origin='upper', cmap='Greys_r')
    plt.show()
    plt.savefig("plots/processed_image_and_mask.png")

    sample_images = train_imgs['input_images'][0:20]
    sample_masks = train_imgs['target_masks'][0:20]

    model = load_model('models/model_keras1.2.2.h5')



    iwant = 3
    pred = model.predict(sd_input_images[iwant:iwant + 1])
    extracted_rings = tmt.template_match_t(pred[0].copy(), minrad=2.) # x coord, y coord, radius

    fig = plt.figure(figsize=[16, 16])
    [[ax1, ax2], [ax3, ax4]] = fig.subplots(2, 2)
    ax1.imshow(sd_input_images[iwant].squeeze(), origin='upper', cmap='Greys_r', vmin=0, vmax=1.1)
    ax2.imshow(sd_target_masks[iwant].squeeze(), origin='upper', cmap='Greys_r')
    ax3.imshow(pred[0], origin='upper', cmap='Greys_r', vmin=0, vmax=1)
    ax4.imshow(sd_input_images[iwant].squeeze(), origin='upper', cmap="Greys_r")
    for x, y, r in extracted_rings:
        circle = plt.Circle((x, y), r, color='blue', fill=False, linewidth=2, alpha=0.5)
        ax4.add_artist(circle)
    ax1.set_title('Moon DEM Image')
    ax2.set_title('Ground-Truth Target Mask')
    ax3.set_title('CNN Predictions')
    ax4.set_title('Post-CNN Craters')
    plt.show()
    plt.savefig("plots/trained_model_results.png")
    

if __name__ == '__main__':
    main()