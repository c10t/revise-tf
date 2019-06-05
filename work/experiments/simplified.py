#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from argparse import ArgumentParser
from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array


def preprocess_image(image_path, img_nrows, img_ncols):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x, img_nrows, img_ncols):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))

    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def main(base_image, style_image, result_prefix):
    width, height = load_img(base_image).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)
    x = preprocess_image(base_image, img_nrows, img_ncols)
    # ...
    # some
    # image
    # processing
    # here
    # ...
    img = deprocess_image(x.copy(), img_nrows, img_ncols)
    fname = result_prefix + '_at_iteration_0.png'
    save_img(fname, img)


if __name__ == '__main__':
    parser = ArgumentParser(description='Neural Style Transfer CLI')
    parser.add_argument(
        'base_image', metavar='base', type=str,
        help='path to the image which will be transformed'
    )
    parser.add_argument(
        'style_image', metavar='ref', type=str,
        help='path to the image for style reference'
    )
    parser.add_argument(
        '--result_prefix', metavar='prefix', type=str,
        help='Prefix for the transformed images',
        default='styled_', required=False
    )
    args = parser.parse_args()

    base_image = args.base_image
    style_image = args.style_image
    result_prefix = args.result_prefix
    main(base_image, style_image, result_prefix)
