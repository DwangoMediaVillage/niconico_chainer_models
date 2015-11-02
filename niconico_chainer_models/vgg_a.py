import chainer
import chainer.optimizers

class VggA(object):
    def __init__(self, outputdim, weight_decay=None, optimizer=None):
        if optimizer is None:
            self.optimizer = chainer.optimizers.AdaGrad(lr=0.001)
        else:
            self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.functions = Functions(outputdim)
        self.optimizer.setup(self.functions)
        self.outputdim = outputdim

    def update_outputdim(self, outputdim):
        self.functions.fc8=chainer.functions.Linear(4096, outputdim)
        self.optimizer.setup(self.functions)

    def train_multi(self, x_data, y_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data, volatile=False)
        h = self.functions.forward(x)[0]
        self.optimizer.zero_grads()
        error = chainer.functions.mean_squared_error(h, y)
        error.backward()
        self.optimizer.update()
        return error.data

    def predict(self, x_data):
        x = chainer.Variable(x_data)
        return self.functions.forward(x, train=False)[0].data

    def predict_all(self, x_data):
        x = chainer.Variable(x_data)
        return self.functions.forward(x, train=False)[1]

    def to_gpu(self):
        self.functions.to_gpu()
        self.optimizer.setup(self.functions)

class Functions(chainer.FunctionSet):
    def __init__(self, outputdim):
        super(Functions, self).__init__(
            conv1_1=chainer.functions.Convolution2D(3, 64, 3, stride=1, pad=1),

            conv2_1=chainer.functions.Convolution2D(64, 128, 3, stride=1, pad=1),

            conv3_1=chainer.functions.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=chainer.functions.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=chainer.functions.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=chainer.functions.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=chainer.functions.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=chainer.functions.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6=chainer.functions.Linear(25088, 4096),
            fc7=chainer.functions.Linear(4096, 4096),
            fc8=chainer.functions.Linear(4096, outputdim) 
        )

    def forward(self, x, train=True):
        x = x

        h1 = chainer.functions.relu(self.conv1_1(x))
        h2 = chainer.functions.max_pooling_2d(h1, 2, stride=2)

        h3 = chainer.functions.relu(self.conv2_1(h2))
        h4 = chainer.functions.max_pooling_2d(h3, 2, stride=2)

        h5 = chainer.functions.relu(self.conv3_1(h4))
        h6 = chainer.functions.relu(self.conv3_2(h5))
        h7 = chainer.functions.max_pooling_2d(h6, 2, stride=2)

        h8 = chainer.functions.relu(self.conv4_1(h7))
        h9 = chainer.functions.relu(self.conv4_2(h8))
        h10 = chainer.functions.max_pooling_2d(h9, 2, stride=2)

        h11 = chainer.functions.relu(self.conv5_1(h10))
        h12 = chainer.functions.relu(self.conv5_2(h11))
        h13 = chainer.functions.max_pooling_2d(h12, 2, stride=2)

        h14 = chainer.functions.dropout(chainer.functions.relu(self.fc6(h13)), train=train, ratio=0.5)
        h15 = chainer.functions.dropout(chainer.functions.relu(self.fc7(h14)), train=train, ratio=0.5)
        h16 = self.fc8(h15)
        h17 = chainer.functions.sigmoid(h16)

        return h17, {
            "x": x,
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "h4": h4,
            "h5": h5,
            "h6": h6,
            "h7": h7,
            "h8": h8,
            "h9": h9,
            "h10": h10,
            "h11": h11,
            "h12": h12,
            "h13": h13,
            "h14": h14,
            "h15": h15,
            "h16": h16,
            "h17": h17
        }
