# -*- coding: utf-8 -*-
# Milestone 1#
import os

img_dir = './tmp/nst'
import wget
import IPython.display

if not os.path.exists(img_dir):
    os.makedirs(img_dir)
    """
    ### Import the content and style images.
    """
wget.download("https://upload.wikimedia.org/wikipedia/commons/d/d7"
              "/Green_Sea_Turtle_grazing_seagrass.jpg", img_dir)
wget.download("https://upload.wikimedia.org/wikipedia/commons/0/0a/"
              "The_Great_Wave_off_Kanagawa.jpg", img_dir)

"""### Import and configure modules"""
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

# Set up some global values here
content_img_path = './tmp/nst/Green_Sea_Turtle_grazing_seagrass.jpg'
stlye_img_path = './tmp/nst/The_Great_Wave_off_Kanagawa.jpg'

"""##Input image"""


def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def imshow(img, title=None):
    # Remove the batch dimension
    out = np.squeeze(img, axis=0)
    # Normalize for display
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)


plt.figure(figsize=(10, 10))

"""## These are the input images : Style Image and Content Image. The objective is to get create a new 'Generated' 
	  image that has the style of the style image, but the content of the content image."""
content = load_img(content_img_path).astype('uint8')
style = load_img(stlye_img_path).astype('uint8')

plt.subplot(1, 2, 1)
imshow(content, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style, 'Style Image')
plt.show()

"""##Preprocessing the input images to the formate accepted by the vgg19 CNN."""


def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


"""In order to view the outputs of our optimization, we are required to perform the inverse preprocessing step. Furthermore, since our optimized image may take its values anywhere between $- \infty$ and $\infty$, we must clip to maintain our values from within the 0-255 range."""


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Content layer to extract the content of the image.
content_layers = ['block5_conv2']

# Layers used to extract the style of the image.
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv2',
                'block5_conv2'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

""" Creates our model with access to intermediate layers.

  This function will load the VGG19 model and access the intermediate layers.
  These layers will then be used to create a new model that will take input image
  and return the outputs from these intermediate layers from the VGG model.

  Returns:
    returns a keras model that takes image inputs and outputs the style and content intermediate layers.
 """


def get_model():
    # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model
    return models.Model(vgg.input, model_outputs)


"""
### Calculating the content loss
	We calculate the content loss my taking the mean squared error between the activation of the content image when passed into VGG and probed at an itermediate layer	
	and the activation out of the generated or base image when passed and probed at the same intermediate later. We basically perform gradient descent to minimize the
	error between these 2 outputs.
"""


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


"""## Calculating the style Loss
We represent style using gram matrixes that denote the correlation between activations of any combination of channels.

To generate a style for our base input image, we perform gradient descent from the content image to transform it into an image that matches the style representation of the original image. We do so by minimizing the mean squared distance between the feature correlation map of the style image and the input image. The contribution of each layer to the total style loss is described by

			E_l = \frac{1}{4N_l^2M_l^2} \sum_{i,j}(G^l_{ij} - A^l_{ij})^2

where $G^l_{ij}$ and $A^l_{ij}$ are the respective style representation in layer $l$ of $x$ and $a$. $N_l$ describes the number of feature maps, each of size $M_l = height * width$. Thus, the total style loss across each layer is
 
			L_{style}(a, x) = \sum_{l \in L} w_l E_l

where we weight the contribution of each layer's loss by some factor $w_l$. In our case, we weight each layer equally (w_l =\frac{1}{|L|})
"""


def gram_matrix(input_tensor):
    # We make the image channels first
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    # with tf.device("/gpu:2"):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(
        tf.square(gram_style - gram_target))  # / (4. * (channels ** 2) * (width * height) ** 2)


"""## Apply style transfer to our images
	Function to compute content and style feature representations.
	Arguments:
		model: The model that we are using.
		content_img_path: Content image path.
		stlye_img_path: Style image path.

	Returns:
		returns Style and Content features.
  """


def get_feature_representations(model, content_img_path, stlye_img_path):
    # Load our images in
    content_image = load_and_process_img(content_img_path)
    style_image = load_and_process_img(stlye_img_path)

    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


"""Funtion to compute the total loss of the image.

  Arguments:
    model: Model to access the intermediate layers.
    loss_weights: style weight, content weight, and total variation weight
    init_image: Initial base image. We update this image by applying gradient descent with respect to
	the loss calculated for this image. We try to reach a minima.
    gram_style_features: Permutations of the gram matrixes calculated based on the correlations of the style layers.
    content_features: Permutations of the outputs defined for the content image.

  Returns:
    returns the total loss, style loss, content loss, and total variational loss
"""


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    # Feeding the input image to our model.
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulating style loss from all the layers and equally weighting contribution of each loss layer.
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # Accumulating content loss from all the layers.
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0],
                                                                     target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Total loss
    loss = style_score + content_score
    return loss, style_score, content_score


"""Then computing the gradients is easy:"""


def compute_gradients(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    # Compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


"""### Running style transfer"""


def run_style_transfer(content_img_path,
                       stlye_img_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    # This is a pretrained model, so we do not need to perform any sort of training using any dataset. Setting trainable = false.
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Getting the style and feature representations of the image from the corresponding intermediate layers.
    style_features, content_features = get_feature_representations(model, content_img_path,
                                                                   stlye_img_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = load_and_process_img(content_img_path)
    init_image = tfe.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

    iter_count = 1

    # Store the best result.
    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # Display
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations / (num_rows * num_cols)
    start_time = time.time()
    global_start = time.time()

    # Weights for VGG19.
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_gradients(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()

        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        if i % display_interval == 0:
            start_time = time.time()
            plot_img = init_image.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            IPython.display.clear_output(wait=True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}, '
                  'time: {:.4f}s'.format(loss, style_score, content_score,
                                         time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14, 4))
    for i, img in enumerate(imgs):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    return best_img, best_loss


best, best_loss = run_style_transfer(content_img_path,
                                     stlye_img_path, num_iterations=1000)

Image.fromarray(best)

"""## Display the output (best image)"""


def display_results(best_img, content_img_path, stlye_img_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_img_path)
    style = load_img(stlye_img_path)

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')

    if show_large_final:
        plt.figure(figsize=(10, 10))

        plt.imshow(best_img)
        plt.title('Output Image')
        plt.show()


display_results(best, content_img_path, stlye_img_path)

# To be contd... Milestone2: Photo realistic style transfer.
# Idea so far :
# 1. Image segmentation (divide image into segments and label each segment)
# 2. Along with style, extract the pixel values of the each respective segment in the image.
# 3. Transfer these values to the base image using gradient descent.
