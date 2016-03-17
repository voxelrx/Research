import os
import time
import theano
import network
import numpy as np
import cPickle as Pickle
from utilities import scaling
from theano import tensor as t
from scipy.io import loadmat, savemat
from utilities.RSA import rsa, multidimensional_scaling, agglomerative_clustering


def data_read(filename):

    print "loading data..."
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", filename)
    data = loadmat(file_path)['Data']
    assert data is not None

    return data


def data_label_read(filename):

    print "loading data..."
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "data", filename)
    data = loadmat(file_path)['labels']
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
    data = scaling.lcn_3d_input(data, kernel_shape=[3, 9, 9], n_maps=data.shape[2])

    return data


def s3_connect():

    import boto
    from boto.s3.key import Key

    s3 = boto.connect_s3()
    my_bucket = 'voxelrx'
    bucket = s3.get_bucket(my_bucket)
    k = Key(bucket)

    return k


def create_directory():

    directory_format = "./saved/%4d-%02d-%02d_%02dh%02dm%02ds"
    directory_name = directory_format % time.localtime()[0:6]
    os.mkdir(directory_name)

    return directory_name


def save_model(model, k, directory_name):

    full_path = directory_name + '/model.pkl'
    Pickle.dump(model, open(full_path, 'w'), Pickle.HIGHEST_PROTOCOL)
    k.key = full_path
    k.set_contents_from_filename(full_path)
    os.remove(full_path)


def save_activations(net_activations, k, directory_name):

    full_path = directory_name + '/data_lcn.mat'
    savemat(full_path, net_activations)
    k.key = full_path
    k.set_contents_from_filename(full_path)
    os.remove(full_path)


def main():

    # load in the data, shuffle, and preprocess
    data = data_read('data.mat')
    data = shuffle_data(data)[0]
    data = preprocess_data(
        data.reshape(
            data.shape[0],
            data.shape[1],
            1,
            data.shape[2],
            data.shape[3]
        )
    )

    # create sub-folder for saved model
    key = s3_connect()
    directory = create_directory()
    x = {'data_lcn3d': data}
    save_activations(x, key, directory)

if __name__ == '__main__':
    main()
