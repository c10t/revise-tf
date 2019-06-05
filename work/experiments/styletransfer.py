#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from argparse import ArgumentParser
from time import time
from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from scipy.optimize import fmin_l_bfgs_b


DESCRIPTION = """
Neural Style Transfer CLI
"""


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


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination, img_nrows, img_ncols):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def total_variation_loss(x, img_nrows, img_ncols):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        left = x[:, :, :img_nrows - 1, :img_ncols - 1]
        a = K.square(left - x[:, :, 1:, :img_ncols - 1])
        b = K.square(left - x[:, :, :img_nrows - 1, 1:])
    else:
        left = x[:, :img_nrows - 1, :img_ncols - 1, :]
        a = K.square(left - x[:, 1:, :img_ncols - 1, :])
        b = K.square(left - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def eval_loss_and_grads(x, f_outputs, img_nrows, img_ncols):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):

    def __init__(self, f_outputs, img_nrows, img_ncols):
        self.loss_value = None
        self.grads_values = None
        self.f_outputs = f_outputs
        self.img_nrows = img_nrows
        self.img_ncols = img_ncols

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(
            x, self.f_outputs, self.img_nrows, self.img_ncols
        )
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def main(base_image, style_image, result_prefix, iterations, weights):
    content_weight, style_weight, total_variation_weight = weights

    width, height = load_img(base_image).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    # get tensor representations of our images
    b_img = K.variable(preprocess_image(base_image, img_nrows, img_ncols))
    s_img = K.variable(preprocess_image(style_image, img_nrows, img_ncols))

    # this will contain our generated image
    if K.image_data_format() == 'channels_first':
        combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
    else:
        combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

    # combine the 3 images into a single Keras tensor
    input_tensor = K.concatenate([b_img, s_img, combination_image], axis=0)

    # build the VGG19 network with our 3 images as input
    # the model will be loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(
        input_tensor=input_tensor, weights='imagenet', include_top=False
    )
    print('Model Loaded.')
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    loss = K.variable(0.0)
    layer_features = outputs_dict['block5_conv2']
    base_img_feats = layer_features[0, :, :, :]
    combination_feats = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_img_feats, combination_feats)

    feature_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]

    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(
            style_reference_features, combination_features,
            img_nrows, img_ncols
        )
        loss += (style_weight / len(feature_layers)) * sl

    loss += total_variation_weight * total_variation_loss(
        combination_image, img_nrows, img_ncols
    )

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combination_image)

    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([combination_image], outputs)
    evaluator = Evaluator(f_outputs, img_nrows, img_ncols)

    # run scipy-based optimization (L-BFGS)
    # over the pixels of the generated image
    # so as to minimize the neural style loss
    x = preprocess_image(base_image, img_nrows, img_ncols)

    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time()
        x, min_val, info = fmin_l_bfgs_b(
            evaluator.loss, x.flatten(),
            fprime=evaluator.grads, maxfun=20
        )
        print('Current loss value:', min_val)
        # save current generated image
        img = deprocess_image(x.copy())
        fname = result_prefix + '_at_iteration_%d.png' % i
        save_img(fname, img)
        end_time = time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))


if __name__ == '__main__':
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        'base_image', metavar='base', type=str,
        help='path to the image which will be transformed'
    )
    parser.add_argument(
        'style_image', metavar='ref', type=str,
        help='path to the image for style reference'
    )
    parser.add_argument(
        'result_prefix', metavar='prefix', type=str,
        help='Prefix for the transformed images',
        default='styled_'
    )
    args = parser.parse_args()

    base_image = args.base_image
    style_image = args.style_image
    result_prefix = args.result_prefix
    iterations = 10
    weights = (0.025, 1.0, 1.0)
    main(base_image, style_image, result_prefix, iterations, weights)
