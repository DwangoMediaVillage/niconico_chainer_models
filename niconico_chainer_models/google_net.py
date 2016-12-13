import chainer
import chainer.links as L
import chainer.functions as F


class GoogLeNet(chainer.Chain):

    """
    .. seealso::

       https://github.com/pfnet/chainer/blob/c67fa7869071c6470fd666be4be7f3c021d507d2/examples/imagenet/googlenetbn.py
    """

    def __init__(self):
        super(GoogLeNet, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, stride=2, pad=3, nobias=True),
            norm1=L.BatchNormalization(64),
            conv2=L.Convolution2D(64, 192, 3, pad=1, nobias=True),
            norm2=L.BatchNormalization(192),
            inc3a=L.InceptionBN(192, 64, 64, 64, 64, 96, 'avg', 32),
            inc3b=L.InceptionBN(256, 64, 64, 96, 64, 96, 'avg', 64),
            inc3c=L.InceptionBN(320, 0, 128, 160, 64, 96, 'max', stride=2),
            inc4a=L.InceptionBN(576, 224, 64, 96, 96, 128, 'avg', 128),
            inc4b=L.InceptionBN(576, 192, 96, 128, 96, 128, 'avg', 128),
            inc4c=L.InceptionBN(576, 128, 128, 160, 128, 160, 'avg', 128),
            inc4d=L.InceptionBN(576, 64, 128, 192, 160, 192, 'avg', 128),
            inc4e=L.InceptionBN(576, 0, 128, 192, 192, 256, 'max', stride=2),
            inc5a=L.InceptionBN(1024, 352, 192, 320, 160, 224, 'avg', 128),
            inc5b=L.InceptionBN(1024, 352, 192, 320, 192, 224, 'max', 128),
            out_tag=L.Linear(1024 + 8, 3000),

            conva=L.Convolution2D(576, 128, 1, nobias=True),
            norma=L.BatchNormalization(128),
            lina=L.Linear(2048, 1024, nobias=True),
            norma2=L.BatchNormalization(1024),
            out_a_tag=L.Linear(1024 + 8, 3000),

            convb=L.Convolution2D(576, 128, 1, nobias=True),
            normb=L.BatchNormalization(128),
            linb=L.Linear(2048, 1024, nobias=True),
            normb2=L.BatchNormalization(1024),
            out_b_tag=L.Linear(1024 + 8, 3000),
        )

    def forward(self, x, z, test=True):

        self.inc3a.train = not test
        self.inc3b.train = not test
        self.inc3c.train = not test
        self.inc4a.train = not test
        self.inc4b.train = not test
        self.inc4c.train = not test
        self.inc4d.train = not test
        self.inc4e.train = not test
        self.inc5a.train = not test
        self.inc5b.train = not test

        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x), test=test)), 3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h), test=test)), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = F.relu(self.norma(self.conva(a), test=test))
        a = F.relu(self.norma2(self.lina(a), test=test))
        a = F.concat((a, z))

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = F.relu(self.normb(self.convb(b), test=test))
        b = F.relu(self.normb2(self.linb(b), test=test))
        b = F.concat((b, z))

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = F.reshape(h, (h.data.shape[0], 1024))
        h = F.concat((h, z))

        return a, b, h

    def tag(self, x, z, test=True):
        a, b, h = self.forward(x, z, test=test)
        tag_a = F.sigmoid(self.out_a_tag(a))
        tag_b = F.sigmoid(self.out_b_tag(b))
        tag = F.sigmoid(self.out_tag(h))
        tag = tag * 0.8 + tag_a * 0.1 + tag_b * 0.1
        return tag
