import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn

class Model(object):
    def __init__(self, ctx='cpu', file=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()
        
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(32, kernel_size=(3,3), padding=(1,1), activation='relu'),
                nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),

                nn.Conv2D(64, kernel_size=(3,3), padding=(1,1), activation='relu'),
                nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),

                nn.Conv2D(128, kernel_size=(3,3), padding=(1,1), activation='relu'),
                nn.Conv2D(128, kernel_size=(3,3), padding=(1,1), activation='relu'),
                nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),

                nn.Conv2D(256, kernel_size=(3,3), padding=(1,1), activation='relu'),
                nn.Conv2D(256, kernel_size=(3,3), padding=(1,1), activation='relu'),
                nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),

                nn.Conv2D(256, kernel_size=(3,3), padding=(1,1), activation='relu'),
                nn.Conv2D(256, kernel_size=(3,3), padding=(1,1), activation='relu'),
                nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),

                nn.Flatten(),
                nn.Dense(2048, activation='relu'),
                nn.Dropout(.2),
                nn.Dense(2048, activation='relu'),
                nn.Dropout(.5),
                nn.Dense(10),
            )

        if file != None:
            self.net.load_params(file, ctx=self.ctx)
        else:
            self.net.initialize(ctx=self.ctx, init=mx.init.Xavier())

        self.net.hybridize()

    def train(self, batches, lr):
        trainer = mx.gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': lr})
        loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        total_L = 0
        for x, y in batches:
            x, y = nd.array(x, self.ctx), nd.array(y, self.ctx)
            with mx.autograd.record():
                p = self.net(x)
                L = loss(p, y).mean()
            L.backward()
            total_L += L.asscalar() / len(batches)
            trainer.step(1)
        return total_L

    def predict(self, x):
        return self.net(nd.array(x, self.ctx)).argmax(axis=1).asnumpy()

    def save(self, file):
        return self.net.save_params(file)