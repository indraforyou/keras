import numpy as np
import matplotlib.pyplot as plt
import urllib
import io
# import skimage.transform
import cv2

import keras
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.applications import vgg16
from keras import backend as K


class ModifiedBackprop(object):

    def __init__(self, nonlinearity):
        self.nonlinearity = nonlinearity
        self.ops = {}  # memoizes an OpFromGraph instance per tensor type

    def __call__(self, x):
        # OpFromGraph is oblique to Theano optimizations, so we need to move
        # things to GPU ourselves if needed.
        if theano.sandbox.cuda.cuda_enabled:
            maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
        else:
            maybe_to_gpu = lambda x: x
        # We move the input to GPU if needed.
        x = maybe_to_gpu(x)
        # We note the tensor type of the input variable to the nonlinearity
        # (mainly dimensionality and dtype); we need to create a fitting Op.
        tensor_type = x.type
        # If we did not create a suitable Op yet, this is the time to do so.
        if tensor_type not in self.ops:
            # For the graph, we create an input variable of the correct type:
            inp = tensor_type()
            # We pass it through the nonlinearity (and move to GPU if needed).
            outp = maybe_to_gpu(self.nonlinearity(inp))
            # Then we fix the forward expression...
            op = theano.OpFromGraph([inp], [outp])
            # ...and replace the gradient with our own (defined in a subclass).
            op.grad = self.grad
            # Finally, we memoize the new Op
            self.ops[tensor_type] = op
        # And apply the memoized Op to the input we got.
        return self.ops[tensor_type](x)

class GuidedBackprop(ModifiedBackprop):
    def grad(self, inputs, out_grads):
        (inp,) = inputs
        (grd,) = out_grads
        dtype = inp.dtype
        return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)

modded_relu = GuidedBackprop(keras.activations.relu)  # important: only instantiate this once!
for layer in model.layers:
    if 'activation' in layer.get_config() and layer.get_config()['activation'] == 'relu':
        layer.activation = modded_relu
        # layer.activation = theano.function([],[])


# def load_img(path, grayscale=False, target_size=None):
#     # img = cv2.imread(path, grayscale)
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     if target_size:
#         img_ch = cv2.resize(img, (target_size[1], target_size[0]))
#     img_ch = img_ch.astype(np.float32, copy=True)
#     mean_pixel = [103.939, 116.779, 123.68]
#     for c in range(3):
#         img_ch[:, :, c] = img_ch[:, :, c] - mean_pixel[c]

#     exit()
#     return img

def compile_saliency_function(model):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    """
    inp = model.input
    outp = model.output
    max_outp = K.max(outp, axis=1)
    sum_max_outp = K.sum(max_outp)
    saliency = K.gradients(sum_max_outp, inp)
    max_class = K.argmax(outp, axis=1)
    # print inp
    # print saliency
    # print max_class
    return K.function([inp], saliency+[max_class])

def prepare_image(fname):
    img_original = cv2.imread(fname)
    img_org = img_original.astype(np.float32, copy=True)
    img = cv2.resize(img_original, (224, 224))

    mean_pixel = [103.939, 116.779, 123.68]
    img = img.astype(np.float32, copy=False)
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]
    # img = img.transpose((2,0,1))

    img = np.expand_dims(img, axis=0)
    return img_original, img

def show_images(img_original, saliency, max_class, title):
    print saliency.shape
    # get out the first map and class from the mini-batch
    saliency = saliency[0]
    print saliency.shape
    # saliency = saliency[::-1].transpose(1, 2, 0)
    print saliency.shape
    max_class = max_class[0]
    # plot the original image and the three saliency map variants
    plt.figure(figsize=(10, 10), facecolor='w')
    # plt.suptitle("Class: " + classes[max_class] + ". Saliency: " + title)
    plt.suptitle("Class: " + str(max_class) + ". Saliency: " + title)
    plt.subplot(2, 2, 1)
    plt.title('input')
    plt.imshow(img_original)
    plt.subplot(2, 2, 2)
    plt.title('abs. saliency')
    x = np.abs(saliency).max(axis=-1)
    print x.shape
    plt.imshow(x, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('pos. saliency')
    x = (np.maximum(0, saliency) / saliency.max())[:,:,0]
    print x.shape
    plt.imshow(x)
    plt.subplot(2, 2, 4)
    plt.title('neg. saliency')
    x = (np.maximum(0, -saliency) / -saliency.min())[:,:,0]
    print x.shape
    plt.imshow(x)
    plt.show()

fname = '/home/isur2/Downloads/test.jpg'
# load_img(fname, target_size=(224, 224))
img_original, img = prepare_image(fname)

model = vgg16.VGG16(weights='imagenet', include_top=True)

saliency_fn = compile_saliency_function(model)
saliency, max_class = saliency_fn([img])
# plt.figure(figsize=(10, 10), facecolor='w')
# plt.imshow(img_original)
# plt.figure(figsize=(10, 10), facecolor='w')
# plt.imshow(np.squeeze(saliency))

# plt.show()

show_images(img_original, saliency, max_class, "default gradient")








