# My insane experiment: use bidirectional rnn to replace convolutions layer

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn
import mxnet.gluon.rnn as rnn

class BiLSTM(nn.Block):
    def __init__(self, nhidden, **kwargs):
        super(BiLSTM, self).__init__(**kwargs)

        with self.name_scope():
            self.hcell = rnn.BidirectionalCell(
                rnn.LSTMCell(nhidden),
                rnn.LSTMCell(nhidden),
            )
            self.vcell = rnn.BidirectionalCell(
                rnn.LSTMCell(nhidden),
                rnn.LSTMCell(nhidden),
            )

    def forward(self, x): # NCHW
        h, w = x.shape[2], x.shape[3]
        res = []
        for i in range(h):
            res.append(nd.stack(*self.hcell.unroll(w, x[:, :, i, :], layout='NCT')[0], axis=2)) # NCW
        for i in range(w):
            res.append(nd.stack(*self.vcell.unroll(h, x[:, :, :, i], layout='NCT')[0], axis=2)) # NCH
        res = nd.relu(nd.stack(*res[:h], axis=2) + nd.stack(*res[h:], axis=3))
        return nd.concat(x, res, dim=1)

class Model(object):
    def __init__(self, ctx='cpu', file=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()

        self.net = nn.Sequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(32, kernel_size=(5,5), padding=(2,2), activation='relu'),
                nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),

                BiLSTM(16),
                BiLSTM(16),
                nn.Conv2D(32, kernel_size=(1,1), activation='relu'),
                nn.GlobalAvgPool2D(),
                
                nn.Flatten(),
                nn.Dense(10)
            )

        if file != None:
            self.net.load_params(file, ctx=self.ctx)
        else:
            self.net.initialize(ctx=self.ctx)

        self.trainer = mx.gluon.Trainer(self.net.collect_params(), 'adam')

    def train(self, batches):
        loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        total_L = 0
        for x, y in batches:
            x, y = nd.array(x, self.ctx), nd.array(y, self.ctx)
            with mx.autograd.record():
                p = self.net(x)
                L = loss(p, y).mean()
            L.backward()
            total_L += L.asscalar() / len(batches)
            self.trainer.step(1)
        return total_L

    def predict(self, x):
        return self.net(nd.array(x, self.ctx)).argmax(axis=1).asnumpy()

    def save(self, file):
        return self.net.save_params(file)