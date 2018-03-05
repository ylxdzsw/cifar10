import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn

class Inception(nn.HybridBlock):
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception, self).__init__(**kwargs)

        with self.name_scope():
            self.p1_conv1 = nn.Conv2D(n1_1, kernel_size=(1, 1), activation='relu')
            self.p2_conv1 = nn.Conv2D(n2_1, kernel_size=(1, 1), activation='relu')
            self.p2_conv3 = nn.Conv2D(n2_3, kernel_size=(3, 3), padding=(1, 1), activation='relu')
            self.p3_conv1 = nn.Conv2D(n3_1, kernel_size=(1, 1), activation='relu')
            self.p3_conv5 = nn.Conv2D(n3_5, kernel_size=(5, 5), padding=(2, 2), activation='relu')
            self.p4_pool3 = nn.MaxPool2D(pool_size=(3, 3), padding=(1, 1), strides=(1, 1))
            self.p4_conv1 = nn.Conv2D(n4_1, kernel_size=(3, 3), padding=(1, 1), activation='relu')

    def hybrid_forward(self, F, x):
        p1 = self.p1_conv1(x)
        p2 = self.p2_conv3(self.p2_conv1(x))
        p3 = self.p3_conv5(self.p3_conv1(x))
        p4 = self.p4_conv1(self.p4_pool3(x))
        return F.concat(p1, p2, p3, p4, dim=1)

class Model(object):
    def __init__(self, ctx='cpu', file=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()
        
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(32, kernel_size=(7,7), padding=(3,3), activation='relu'),

                Inception(32, 48, 64, 8, 16, 16),
                Inception(64, 64, 96, 16, 48, 32),
                nn.MaxPool2D(pool_size=(3,3), strides=(2,2)),

                Inception(64, 48, 104, 8, 24, 32),
                Inception(64, 64, 128, 12, 32, 32),
                nn.MaxPool2D(pool_size=(3,3), strides=(2,2)),

                Inception(64, 64, 96, 16, 48, 32),
                Inception(32, 48, 64, 8, 16, 16),
                nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),

                nn.Flatten(),
                nn.Dense(10)
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