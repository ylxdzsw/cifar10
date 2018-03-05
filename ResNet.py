import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn

class Residual(nn.HybridBlock):
    def __init__(self, nkernel, down_sample=False, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.down_sample = down_sample
        strides = 2 if down_sample else 1
        with self.name_scope():
            self.conv1 = nn.Conv2D(nkernel, kernel_size=(3, 3), padding=(1, 1), strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(nkernel, kernel_size=(3, 3), padding=(1, 1))
            self.bn2 = nn.BatchNorm()
            if down_sample:
                self.conv3 = nn.Conv2D(nkernel, kernel_size=(1, 1), strides=strides)
            
    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down_sample:
            x = self.conv3(x)
        return F.relu(out + x)

class Model(object):
    def __init__(self, ctx='cpu', file=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()
        
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(32, kernel_size=(3,3), padding=(1,1)),
                nn.BatchNorm(),
                nn.Activation('relu'),

                Residual(32),
                Residual(32),
                Residual(32),

                Residual(64, True),
                Residual(64),
                Residual(64),

                Residual(128, True),
                Residual(128),
                Residual(128),
                
                nn.GlobalAvgPool2D(),
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