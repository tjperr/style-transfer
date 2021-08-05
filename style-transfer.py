import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from tqdm import tqdm
import time

base_image_path = "flower_me/flower_me_low_content_loss_at_iteration_0.png"
#base_image_path = "IMG_3264.jpg"
#base_image_path = "flower_me/flower_me_at_iteration_200.png"

#, "https://i.imgur.com/F28w3Ac.jpg")
style_reference_image_path = "IMG_3772.jpg"
#  keras.utils.get_file(
#     "starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg"
# )
result_prefix = "flower_me/flower_me_low_content_loss"

# Weights of the different loss components
total_variation_weight = 1e-6
style_weight = 2e-6
content_weight = 1e-8
col_weight = 2000

# Dimensions of the generated picture.
width, height = keras.preprocessing.image.load_img(base_image_path).size
img_nrows = 600
img_ncols = int(width * img_nrows / height)

def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


# The gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    #print("gram")
    #print(x.shape)
    x = tf.transpose(x, (2, 0, 1))
    #print(x.shape)
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    #print(features.shape)
    gram = tf.matmul(features, tf.transpose(features))
    #print(gram.shape)
    return gram


# The "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


# An auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


# The 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))

def colour_loss(base, combination):
    a = tf.reduce_mean(base, axis=[0,1,2])
    b = tf.reduce_mean(combination, axis=[0,1,2])
    return tf.reduce_sum(tf.square(a - b))

# Build a VGG19 model loaded with pre-trained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Set up a model that returns the activation values for every layer in
# VGG19 (as a dict).
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

# List of layers to use for the style loss.
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# The layer to use for the content loss.
content_layer_name = "block5_conv2"


def compute_loss(combination_image, base_image, style_reference_image):
    #print('compute loss')
    #print(base_image.shape)
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    #print(input_tensor.shape)
    features = feature_extractor(input_tensor)
    #print(features)
    # Initialize the loss
    vl = tf.zeros(shape=())
    sl = tf.zeros(shape=())
    cl = tf.zeros(shape=())
    col_l = tf.zeros(shape=())
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    #print(layer_features.shape)
    #print(layer_features)
    base_image_features = layer_features[0, :, :, :]
    #print(base_image_features.shape)
    combination_features = layer_features[2, :, :, :]
    #print(combination_features.shape)
    cl = content_weight * content_loss(
        base_image_features, combination_features
    )
    print(cl)
    # Add style loss

    for layer_name in style_layer_names:
        #print('style layer loop')
        layer_features = features[layer_name]
        #print(layer_features.shape)
        style_reference_features = layer_features[1, :, :, :]
        #print(style_reference_features.shape)
        combination_features = layer_features[2, :, :, :]
        #print(combination_features.shape)
        sl += (style_weight / len(style_layer_names)) * style_loss(style_reference_features, combination_features)
    print(sl)

    # Add total variation loss
    vl = total_variation_weight * total_variation_loss(combination_image)
    print(vl)

    # Add color loss
    col_l = col_weight * colour_loss(base_image, combination_image)
    print(col_l)
    return cl + sl + vl + col_l

@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.98
    )
)

##print('preprocessing images')
base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

iterations = 400
##print('iterating')
for i in tqdm(range(200, iterations + 1)):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    print("Iteration %d: loss=%.2f" % (i, loss))

    if i % 200 == 0:
        fname = result_prefix + "_at_iteration_%d.png" % i
    img = deprocess_image(combination_image.numpy())
    keras.preprocessing.image.save_img(fname, img)