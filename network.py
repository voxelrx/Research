import theano
import utilities.BP as BP
from theano import tensor as t
from utilities.scaling import lcn_lacombe, lcn_3d
from utilities.init import init_weights
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet.conv3d2d import conv3d
from utilities.maxpool3d import max_pool_3d


def norm(f):

    """
    Perform sparse filtering normalization procedure.

    Parameters:
    ----------
    f : ndarray
        The activation of the network. [neurons x examples]

    Returns:
    -------
    f_hat : ndarray
        The row and column normalized matrix of activation.
        :param f:
    """

    fs = t.sqrt(f ** 2 + 1e-8)                              # soft-max function
    # if fs.shape[1] == 1:
    #     nfs = fs
    # else:
    #     l2fs = t.sqrt(t.sum(fs ** 2, axis=1))               # l2 norm of row
    #     nfs = fs / l2fs.dimshuffle(0, 'x')                  # normalize rows
    nfs = fs
    l2fn = t.sqrt(t.sum(nfs ** 2, axis=0))                  # l2 norm of column
    f_hat = nfs / l2fn.dimshuffle('x', 0)                   # normalize columns

    return f_hat


def convolutional_norm(f):

    """
    Perform convolutional sparse filtering normalization procedure.

    Parameters:
    ----------
    f : ndarray
        The activation of the network. [examples x neurons x dim x dim]

    Returns:
    -------
    f_hat : ndarray
        The row and column normalized matrix of activation.
        :param f:
    """

    fs = t.sqrt(f ** 2 + 1e-8)                              # soft-max function
    l2fs = t.sqrt(t.sum(fs ** 2, axis=[0, 2, 3]))           # l2 norm of example dimensions
    nfs = fs / l2fs.dimshuffle('x', 0, 'x', 'x')            # normalize non-example dimensions
    l2fn = t.sqrt(t.sum(nfs ** 2, axis=[1]))                # l2 norm of neuron dimension
    f_hat = nfs / l2fn.dimshuffle(0, 'x', 1, 2)             # normalize non-neuron dimensions

    return f_hat


def convolutional_3d_norm(f):

    """
    Perform convolutional sparse filtering normalization procedure.

    Parameters:
    ----------
    f : ndarray
        The activation of the network. [examples, depth, filters, height, width]

    Returns:
    -------
    f_hat : ndarray
        The row and column normalized matrix of activation.
        :param f: [examples, depth, filters, height, width]
    """

    fs = t.sqrt(f ** 2 + 1e-8)                              # soft-max function
    l2fs = t.sqrt(t.sum(fs ** 2, axis=[0, 1, 3, 4]))        # l2 norm of example dimensions
    nfs = fs / l2fs.dimshuffle('x', 'x', 0, 'x', 'x')       # normalize non-example dimensions
    l2fn = t.sqrt(t.sum(nfs ** 2, axis=[2]))                # l2 norm of neuron dimension
    f_hat = nfs / l2fn.dimshuffle(0, 1, 'x', 2, 3)          # normalize non-neuron dimensions

    return f_hat


class SparseFilter(object):

    """ Sparse Filtering """

    def __init__(self, w, x):

        """
        Build a sparse filtering model.

        Parameters:
        ----------
        w : ndarray
            Weight matrix randomly initialized.
        x : ndarray (symbolic Theano variable)
            Data for model.
        """

        # assign inputs to sparse filter
        self.w = w
        self.x = x

        # define normalization procedure
        self.norm = norm

    def dot(self):

        """ Returns dot product of weights and input data """

        # f = t.dot(self.w, self.x)
        f = t.dot(self.w, self.x.T)

        return f

    def feed_forward(self):

        """ Performs sparse filtering normalization procedure """

        f_hat = self.norm(self.dot())

        return f_hat

    def criterion(self):

        """ Returns the criterion for model evaluation """

        return self.feed_forward()


class ConvolutionalSF(object):

    """ Convolutional Sparse Filtering """

    def __init__(self, w, x):

        """
        Build a convolutional sparse filtering model.

        Parameters:
        ----------
        w : ndarray
            Weight matrix randomly initialized.
        x : ndarray (symbolic Theano variable)
            Data for model.
        """

        # assign inputs to sparse filter
        self.w = w
        self.x = x

        # define normalization procedure
        self.norm = convolutional_norm

    def dot(self):

        """ Convolve input with model weights """

        f = conv2d(self.x, self.w, subsample=(1, 1))

        return f

    def feed_forward(self):

        """ Performs convolutional sparse filtering procedure """

        f_hat = self.norm(self.dot())

        return f_hat

    def criterion(self):

        """ Returns the criterion for model evaluation """

        return self.feed_forward()

    def max_pool(self):

        """ Perform 2D max pooling """

        return max_pool_2d(self.feed_forward(), ds=(2, 2), ignore_border=True, mode='sum')


class ConvolutionalSF3D(object):

    """ Convolutional Sparse Filtering """

    def __init__(self, w, x, p):

        """
        Build a convolutional sparse filtering model.

        Parameters:
        ----------
        w : ndarray
            Weight matrix randomly initialized.
        x : ndarray (symbolic Theano variable)
            Data for model.
        """

        # assign inputs to sparse filter
        self.w = w
        self.x = x
        self.p = p

        # define normalization procedure
        self.norm = convolutional_3d_norm

    def dot(self):

        """ Convolve input with model weights """

        f = conv3d(self.x, self.w)

        return f

    def feed_forward(self):

        """ Performs convolutional sparse filtering procedure """

        f_hat = self.norm(self.dot())

        return f_hat

    def criterion(self):

        """ Returns the criterion for model evaluation """

        return self.feed_forward()

    def max_pool(self):

        """ Perform 2D max pooling """

        return max_pool_3d(self.feed_forward(), ds=(self.p[0], self.p[1], self.p[2]))


class Layer(object):

    """ Layer object within network """

    def __init__(self, model_type='SparseFilter', weight_dims=(100, 256), layer_input=None,
                 lr=0.0001, dim=2, weights=None, p=None):

        """
        Builds a layer for the network by constructing a model.

        Parameters:
        ----------
        model_type : str
            The model type to build into a given layer.
        weight_dims : list of tuples
            The dimensions of the weight matrices for each layer.
            fully connected: [neurons x input_dim ^ 2]
            convolutional: [filters x dim x dim]
        layer_input : ndarray (symbolic Theano variable)
            The input to a given layer.
        p : int
            The pooling size (assumed to be square).
        group_size : int
            The group size for group sparse filtering.
        step : int
            The step size for group sparse filtering.
        lr : int
            The learning rate for gradient descent.
        """

        # assign network inputs to layer
        self.m = model_type
        self.weight_dims = weight_dims
        self.x = layer_input
        self.lr = lr
        self.dim = dim
        self.p = p

        if weights is None:
            self.w = init_weights(weight_dims)
        elif weights is not None:
            self.w = weights

        # build model based on model_type
        self.model = None
        if model_type == 'SparseFilter':
            self.model = SparseFilter(self.w, self.x)
        elif model_type == 'ConvolutionalSF':
            self.model = ConvolutionalSF(self.w, self.x)
        elif model_type == 'ConvolutionalSF3D':
            self.model = ConvolutionalSF3D(self.w, self.x, self.p)
        assert self.model is not None

    def feed_forward(self):

        """ Feed-forward through the network """

        f_hat = self.model.feed_forward()

        return f_hat

    def max_pool(self):

        """ Return max-pooled output """

        if self.dim == 4 or self.dim == 5:
            pooled = self.model.max_pool()
        else:
            pooled = None

        return pooled

    def criterion(self):

        """ Return the criterion for model evaluation """

        return self.model.criterion()

    def get_cost_updates(self):

        """ Returns the cost and updates for the layer """

        # cost = t.sum(t.abs_(self.criterion()))
        cost = t.sum(t.abs_(t.log(self.criterion() + 1)))
        updates = BP.RMSprop(cost, self.w, lr=self.lr)
        # updates = BP.censor_updates(updates, self.c)        # todo: censor updates for 3d convolution

        return cost, updates

    def get_weights(self):

        """ Returns the weights of the layer """

        weights = self.w

        return weights

    def get_activations(self):

        # activation = self.model.dot()
        activation = t.maximum(0, self.model.dot())

        return activation


class Network(object):

    """ Neural network architecture """

    def __init__(self, model_type='SparseFilter', weight_dims=([100, 256], []),
                 lr=([0.0001], []), opt='GD', dim=([2], []), test='n', batch_size=1000,
                 random='n', weights=None, lcn_kernel=None, lcn_type=None, pool=None):

        """
        Neural network constructor. Defines a network architecture that builds
        layers, each with own model.

        Parameters:
        ----------
        model_type : str
            The model type to build into a given layer.
        weight_dims : list of tuples
            The dimensions of the weight matrices for each layer.
            fully connected: [neurons x input_dim ^ 2]
            convolutional: [filters x dim x dim]
        p : int
            The pooling size (assumed to be square).
        group_size : int
            The group size for group sparse filtering.
        step : int
            The step size for group sparse filtering.
        lr : int
            The learning rate for gradient descent.
        opt : str
            The optimization algorithm used for learning.
        c : str
            Indicates whether the network is fully connected or convolutional.
        """

        # assign the inputs to the network
        self.model = model_type
        self.layers = []
        self.n_layers = len(weight_dims)
        self.opt = opt
        self.weight_dims = weight_dims
        self.dim = dim
        self.test = test
        self.batch_size = batch_size
        self.random_batch = random

        # make assertions
        assert self.n_layers > 0

        # define symbolic variable for input data based on network type
        if self.dim[0] == 2:
            self.x = t.fmatrix('x')
        elif self.dim[0] == 4:
            self.x = t.ftensor4('x')
        elif self.dim[0] == 5:
            ftensor5 = t.TensorType('float32', [False] * 5)
            self.x = ftensor5('x')

        # for each layer, create a layer object
        for l in xrange(self.n_layers):

            print "...for layer %d" % l

            if l == 0:                                                      # first layer
                layer_input = self.x

            else:                                                           # subsequent layers
                # if self.dim[l] == 2:
                #     layer_input = self.layers[l - 1].feed_forward()
                # else:                                                       # i.e., convolutional
                #     # layer_input = self.layers[l - 1].max_pool()
                layer_input = self.layers[l - 1].feed_forward()

                # perform pooling
                if pool[l - 1] is not None:
                    if len(pool[l - 1]) == 2:
                        layer_input = max_pool_2d(
                            layer_input, ds=(pool[l - 1][0], pool[l - 1][1]),
                            ignore_border=True,
                            mode='max'
                        )
                    if len(pool[l - 1]) == 3:
                        layer_input = max_pool_3d(
                            layer_input.dimshuffle((0, 2, 1, 3, 4)),
                            ds=(pool[l - 1][0], pool[l - 1][1], pool[l - 1][2])
                        )

                # perform LCN
                if lcn_type[l - 1] is not None:
                    if lcn_type[l - 1] == '2d':
                        if lcn_kernel is None:
                            layer_input = lcn_lacombe(
                                layer_input,
                                kernel_shape=5,
                                n_maps=self.layers[l - 1].weight_dims[0]
                            )
                        else:
                            layer_input = lcn_lacombe(
                                layer_input,
                                kernel_shape=lcn_kernel[l - 1],
                                n_maps=self.layers[l - 1].weight_dims[0]
                            )
                    elif lcn_type[l - 1] == '3d':
                        if lcn_kernel is None:
                            layer_input = lcn_3d(
                                layer_input.dimshuffle((0, 2, 1, 3, 4)),
                                kernel_shape=5,
                                n_maps=self.layers[l - 1].weight_dims[0]
                            )
                        else:
                            layer_input = lcn_3d(
                                layer_input.dimshuffle((0, 2, 1, 3, 4)),
                                kernel_shape=lcn_kernel[l - 1],
                                n_maps=self.layers[l - 1].weight_dims[0]
                            )

                # shuffle dimensions if only pooling performed
                if pool[l - 1] is not None and lcn_type[l - 1] is None:
                    layer_input = layer_input.dimshuffle((0, 2, 1, 3, 4))

                # layer_input = max_pool_2d(layer_input, ds=(2, 2), ignore_border=True, mode='sum')

                # reshape the input based on the dimensionality of the tensors
                if self.dim[l - 1] != self.dim[l]:
                    layer_input = layer_input.reshape((self.batch_size, -1))

                    # perform normalization before fully connected layer
                    layer_input = layer_input / t.sqrt(t.sum(layer_input, axis=[1])).dimshuffle(0, 'x')

            # define layer and append to network layers
            layer_l = Layer(model_type[l], weight_dims[l], layer_input, lr[l], dim[l], weights, pool[l])
            self.layers.append(layer_l)

    def training_functions(self, data):

        """
        Construct training functions for each layer.

        Parameters:
        ----------
        data : ndarray
            Training data for unsupervised feature learning. Can be patches, full
            images, or video.

        Returns:
        -------
        train_fns : list
            List of compiled theano functions for training each layer.
        out_fns : list
            List of compiled theano functions for retrieving important variables.
            :param data:
        """

        # create batches for training
        batch_end = None
        batch_begin = None
        if self.random_batch == 'n':

            # index to a [mini]batch
            index = t.lscalar('index')

            # define beginning and end of a batch given 'index'
            batch_begin = index * self.batch_size
            batch_end = batch_begin + self.batch_size

        else:

            # index to a [mini]batch
            index = t.ivector('index')

        # initialize empty function lists
        train_fns = []
        out_fns = []
        test_fn = []

        # make data a shared theano variable
        data = theano.shared(data)

        # for each layer define a training, output, and test function
        c = 0
        for l in self.layers:

            print "...for layer %d" % c
            c += 1

            # get outputs for theano functions
            w = l.get_weights()
            cost, updates = l.get_cost_updates()
            f_hat = l.feed_forward()

            # get training function
            fn = None
            if self.opt == 'GD':

                # fn = theano.function(
                #     inputs=[self.x],
                #     outputs=[cost, w],
                #     updates=updates,
                #     # givens={
                #     #     self.x: data[batch_begin:batch_end]
                #     # },
                #     on_unused_input='ignore'
                # )

                if self.random_batch == 'n':

                    fn = theano.function(
                        inputs=[index],
                        outputs=[cost, w],
                        updates=updates,
                        givens={
                            self.x: data[batch_begin:batch_end]
                        },
                        on_unused_input='ignore'
                    )

                elif self.random_batch == 'y':

                    fn = theano.function(
                        inputs=[index],
                        outputs=[cost, w],
                        updates=updates,
                        givens={
                            self.x: data[index]  # indexer
                        },
                        on_unused_input='ignore'
                    )

                train_fns.append(fn)

            # get output function
            out = theano.function([self.x], outputs=[f_hat])
            out_fns.append(out)

            # get test function
            if self.test == 'y':
                test = theano.function([self.x], outputs=[f_hat])
                test_fn.append(test)

        return train_fns, out_fns, test_fn
