import os
import time
import theano
import network
import numpy as np
import cPickle as Pickle
from theano import tensor as t
from utilities import scaling
from scipy.io import loadmat, savemat
import theano
import utilities.BP as BP
from theano import tensor as t
from utilities.init import init_weights
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

from theano.tensor.nnet.conv3d2d import conv3d
from utilities import maxpool3d
from utilities.scaling import lcn_lacombe, lcn_3d


def data_read(filename):

    print "loading data..."
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", filename)
    data = loadmat(file_path)['Data']
    assert data is not None

    return data


def shuffle_data(data):

    print "shuffling data..."
    x = t.ftensor4()
    shuffled = x.dimshuffle(3, 2, 0, 1)

    fn = theano.function(
        inputs=[x],
        outputs=[shuffled],
    )

    return fn(data)


def preprocess_data(data):

    print "pre-processing data..."
    data = np.float32(data)
    print data.shape
    data = scaling.lcn_3d_input(data, kernel_shape=[3, 9, 9], n_maps=data.shape[2])
    print 'done!'

    return data


mri_data = data_read('data.mat')
mri_data = shuffle_data(mri_data)
mri_data = mri_data[0]
mri_data = preprocess_data(mri_data.reshape(19, 30, 1, 256, 256))
print mri_data.shape

ftensor5 = t.TensorType('float32', [False] * 5)

X = ftensor5('x')
w1 = init_weights([13, 3, 1, 11, 11])    # out after (1, 4, 4) max-pooling: [examples, 28, filters, 62, 62]
w2 = init_weights([13, 3, 13, 6, 6])     # out after (2, 4, 4) max-pooling: [examples, 13, filters, 15, 15]
w3 = init_weights([13, 3, 13, 4, 4])     # out after (4, 4, 4) max-pooling: [examples, 3, filters, 3, 3]

w4 = init_weights([13, 13 * 3 * 3 * 3])       # out: [examples, filters]

# w4 = init_weights([13, 2, 13, 3, 3])        # out after max-pooling: [examples, 1, filters, 13, 13]
# w5 = init_weights([13, 13, 2, 2])           # out after (4, 4) max-pooling: [examples, filters, 3, 3]
# w6 = init_weights([13, 13 * 3 * 3])


#######################
conv_out1 = conv3d(
    signals=X,
    filters=w1
)

pool_out1 = maxpool3d.max_pool_3d(
    conv_out1.dimshuffle((0, 2, 1, 3, 4)),
    (1, 4, 4)
)

lcn_out1 = lcn_3d(pool_out1.dimshuffle((0, 2, 1, 3, 4)), [3, 5, 5], 13)
# [examples, 28, filters, 62, 62]

#######################
conv_out2 = conv3d(
    signals=lcn_out1,
    filters=w2
)

pool_out2 = maxpool3d.max_pool_3d(
    conv_out2.dimshuffle((0, 2, 1, 3, 4)),
    (2, 4, 4)
)

lcn_out2 = lcn_3d(pool_out2.dimshuffle((0, 2, 1, 3, 4)), [3, 3, 3], 13)
# [examples, 13, filters, 15, 15]

#######################
conv_out3 = conv3d(
    signals=lcn_out2,
    filters=w3
)

pool_out3 = maxpool3d.max_pool_3d(
    conv_out3.dimshuffle((0, 2, 1, 3, 4)),
    (4, 4, 4)
)
# [examples, filters, 3, 3, 3]  (last dim-shuffle not performed)

#######################
# (perform normalization here)
fc_out4 = t.dot(pool_out3.reshape((2, -1)), w4.T)


#######################
print 'building model...'
fn = theano.function(inputs=[X], outputs=[fc_out4])


test_out = fn(mri_data[:2])
print test_out[0].shape
