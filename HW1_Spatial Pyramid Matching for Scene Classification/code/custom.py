from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import custom_visual_recog
from opts import get_opts


def main():
    opts = get_opts()

    # Q1.2 Compute Dictionary
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    # Q2.1-2.4 Build the recognition system
    n_cpu = util.get_num_CPU()
    custom_visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5 evaluate the recognition system
    n_cpu = util.get_num_CPU()
    conf, accuracy = custom_visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)

    print(conf)
    print(accuracy)
    # np.savetxt(join(opts.out_dir, 'custom_confmat.csv'),
    #            conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'custom_accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()