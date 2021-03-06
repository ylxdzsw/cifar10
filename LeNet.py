import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn

class Model(object):
    def __init__(self, ctx='cpu', file=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()
        
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(20, kernel_size=(5,5), padding=(2,2), activation='relu'),
                nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),
                nn.Conv2D(50, kernel_size=(3,3), padding=(1,1), activation='relu'),
                nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),
                nn.Flatten(),
                nn.Dense(128, activation='relu'),
                nn.Dense(10)
            )

        if file != None:
            self.net.load_params(file, ctx=self.ctx)
        else:
            self.net.initialize(ctx=self.ctx)

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