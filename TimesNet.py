# My insane experiment #2: use multiplication

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn

# baseline 0.576

class Times1(nn.HybridBlock): # 0.582
    def __init__(self, nkernel, **kwargs):
        super(Times1, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(nkernel, kernel_size=(3,3), padding=(1,1))
            self.dense = nn.Dense(nkernel)
            self.pool = nn.GlobalAvgPool2D()

    def hybrid_forward(self, F, x):
        proposed = self.conv(x)
        weight = self.dense(self.pool(proposed))
        weight = weight.expand_dims(axis=-1).expand_dims(axis=-1)
        return F.relu(F.broadcast_mul(proposed, weight))

class Times2(nn.HybridBlock): # 0.598
    def __init__(self, nkernel, **kwargs):
        super(Times2, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(nkernel, kernel_size=(3,3), padding=(1,1))
            self.dense = nn.Dense(nkernel, activation='tanh')
            self.pool = nn.GlobalAvgPool2D()

    def hybrid_forward(self, F, x):
        proposed = self.conv(x)
        weight = self.dense(self.pool(proposed))
        weight = weight.expand_dims(axis=-1).expand_dims(axis=-1)
        return F.relu(F.broadcast_mul(proposed, weight))

class Times3(nn.HybridBlock): # 0.438
    def __init__(self, nkernel, **kwargs):
        super(Times3, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(nkernel, kernel_size=(3,3), padding=(1,1))
            self.dense = nn.Dense(nkernel)
            self.pool = nn.GlobalAvgPool2D()

    def hybrid_forward(self, F, x):
        proposed = self.conv(x)
        weight = F.softmax(self.dense(self.pool(proposed)), axis=1)
        weight = weight.expand_dims(axis=-1).expand_dims(axis=-1)
        return F.relu(F.broadcast_mul(proposed, weight))

class Model(object):
    def __init__(self, ctx='cpu', file=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()
        
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(20, kernel_size=(5,5), padding=(2,2), activation='relu'),
                nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),
                Times3(50),
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