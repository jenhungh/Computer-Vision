import numpy as np
from os.path import join

dictionary = np.load(join('.', 'dictionary.npy'))
print(dictionary)