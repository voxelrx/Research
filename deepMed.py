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


# todo: create indexing function that samples sub-volumes [196, 256, 256] / 5 = [39, 51, 51] (5^3 more filters)

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


def save_to_file(to_save, k, directory_name, filename):

    full_path = os.path.join(directory_name, filename)
    if filename == 'model.pkl':
        Pickle.dump(to_save, open(full_path, 'w'), Pickle.HIGHEST_PROTOCOL)
    else:
        savemat(full_path, to_save)
    k.key = full_path
    k.set_contents_from_filename(full_path)
    os.remove(full_path)


def main():

    # define model parameters
    model_type = [
        'ConvolutionalSF3D',
        'ConvolutionalSF3D',
        'ConvolutionalSF3D',
        'SparseFilter',
    ]
    n_dims = [
        5,
        5,
        5,
        2
    ]
    filename = "data.mat"
    n_filters = [
        50,
        400,
        1600,
        1000
    ]
    dimensions = (
        [n_filters[0], 3, 1, 11, 11],
        [n_filters[1], 3, n_filters[0], 6, 6],
        [n_filters[2], 3, n_filters[1], 4, 4],
        [n_filters[3], n_filters[2] * 3 * 3 * 3]
    )
    learn_rate = [
        0.0001,
        0.00001,
        0.00001,
        0.000001
    ]
    iterations = [
        # 1,
        # 1,
        # 1,
        # 1
        30,
        50,
        20,
        40
    ]
    opt = 'GD'
    test_model = 'y'
    batch_size = 1
    lcn_kernel = [
        [3, 5, 5],
        [3, 3, 3],
        None,
        None
    ]
    lcn_type = [
        '3d',
        '3d',
        None,
        None
    ]
    pool = [
        [1, 4, 4],
        [2, 4, 4],
        [4, 4, 4],
        None
    ]

    # load in the data, shuffle, and preprocess
    data = data_read(filename)
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

    # determine number of batches
    n_batches, rem = divmod(data.shape[0], batch_size)

    # construct the network
    print "building model..."
    model = network.Network(
        model_type=model_type,
        weight_dims=dimensions,
        lr=learn_rate,
        opt=opt,
        dim=n_dims,
        test=test_model,
        batch_size=batch_size,
        random='y',
        weights=None,
        lcn_kernel=lcn_kernel,
        lcn_type=lcn_type,
        pool=pool
    )

    # compile the training, output, and test functions for the network
    print "compiling theano functions..."
    train, outputs, test = model.training_functions(data)

    # train the sparse filtering network
    print "training network..."
    start_time = time.time()
    cost = {}
    weights = {}
    for l in xrange(model.n_layers):

        cost_layer = []
        w = None

        # iterate over training epochs
        for epoch in xrange(iterations[l]):

            # go though [mini]batches
            for batch_index in xrange(n_batches):

                # create index for random [mini]batch
                index = np.int32(np.random.randint(data.shape[0], size=batch_size))

                c, w = train[l](index=index)
                cost_layer.append(c)
                print("Layer %i cost at epoch %i and batch %i: %f" % (l + 1, epoch, batch_index, c))

                # f_hat = outputs[l](data[0].reshape((1, 30, 1, 256, 256)))
                # print f_hat[0].shape

        # add layer cost and weights to the dictionaries
        cost['layer' + str(l)] = cost_layer
        weights['layer' + str(l)] = w

    # calculate and display elapsed training time
    elapsed = time.time() - start_time
    print('Elapsed training time: %f' % elapsed)

    # connect to s3, create sub-folder, and save the model and cost function
    key = s3_connect()
    directory = create_directory()
    save_to_file(model, key, directory, 'model.pkl')
    save_to_file(cost, key, directory, 'cost.mat')

    # get output from last layer of network for each of the MRI scans
    print('Getting network output...')
    network_output = np.zeros((n_filters[-1], data.shape[0]))
    for mri_scan in xrange(data.shape[0]):
        activations = outputs[model.n_layers - 1](data[mri_scan].reshape((1, 30, 1, 256, 256)))[0]
        network_output[:, mri_scan] = activations.reshape((n_filters[-1]))

    net_out = {'net_out': network_output}
    save_to_file(net_out, key, directory, 'activations.mat')

    # # todo: perform pairwise correlations of patient activation spaces from final layer to form RDM
    # print('Performing RSA...')
    # rdm = rsa(network_output)
    #
    # # todo: perform dimensionality reduction using MDS TO SEE IF THERE IS NATURAL CLUSTERING
    # print('Performing dimensionality reduction...')
    # labels = data_label_read('data_labels.mat')
    # multidimensional_scaling(rdm, labels=labels)
    # agglomerative_clustering(rdm, labels=labels)


if __name__ == '__main__':
    main()
