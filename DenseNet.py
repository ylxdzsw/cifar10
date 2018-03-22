import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn

class DenseBlock(nn.HybridBlock):
    def __init__(self, nlayer, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.net = nn.HybridSequential()
            for i in range(nlayer):
                self.net.add(
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    nn.Conv2D(growth_rate, kernel_size=(3, 3), padding=(1, 1))
                )

    def hybrid_forward(self, F, x):
        for layer in self.net:
            out = layer(x)
            x = F.concat(x, out, dim=1)
        return x

class Model(object):
    def __init__(self, ctx='cpu', file=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()
        
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(32, kernel_size=(7,7), padding=(3,3)),

                DenseBlock(4, 8),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(32, kernel_size=(1,1)),
                nn.AvgPool2D(pool_size=(2,2), strides=(2,2)),

                DenseBlock(4, 8),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(32, kernel_size=(1,1)),
                nn.AvgPool2D(pool_size=(2,2), strides=(2,2)),

                DenseBlock(4, 8),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(32, kernel_size=(1,1)),
                nn.AvgPool2D(pool_size=(2,2), strides=(2,2)),

                nn.Flatten(),
                nn.Dense(10)
            )

        if file != None:
            self.net.load_params(file, ctx=self.ctx)
        else:
            self.net.initialize(ctx=self.ctx, init=mx.init.Xavier())

        self.trainer = mx.gluon.Trainer(self.net.collect_params(), 'adam')
        self.net.hybridize()

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