from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorly as tl
from PIL import Image
from scipy.misc import face
from scipy.ndimage import zoom
from tensorly.decomposition import parafac, tucker
from omegaconf import OmegaConf

cfg = OmegaConf.load('conf/config.yml')

random_state = 12345
image = Image.open(cfg.catalogue.sample_path)

image = tl.tensor(zoom(image, (1, 1, 1)), dtype='float64')

def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

# Rank of the CP decomposition
cp_rank = 5 # og 25
weights, factors = parafac(image, rank=cp_rank, init='random', tol=10e-6)
cp_reconstruction = tl.cp_to_tensor((weights, factors))
# Tucker decomposition

tucker_rank = [1, 1, 1]
core, tucker_factors = tucker(image, rank=tucker_rank, init='random', tol=10e-5, random_state=random_state)
tucker_reconstruction = tl.tucker_to_tensor((core, tucker_factors))
(core.size + sum([x.size for x in tucker_factors])) / image.size # only 25%

