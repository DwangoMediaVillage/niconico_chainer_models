import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class GoogLeNet(Chain):

    """
    .. seealso::

       https://github.com/pfnet/chainer/blob/c67fa7869071c6470fd666be4be7f3c021d507d2/examples/imagenet/googlenetbn.py
    """

    def __init__(self):
        super().__init__(
            conv1=F.Convolution2D(3, 64, 7, stride=2, pad=3, nobias=True),
            norm1=F.BatchNormalization(64),
            conv2=F.Convolution2D(64, 192, 3, pad=1, nobias=True),
            norm2=F.BatchNormalization(192),
            inc3a=F.InceptionBN(192, 64, 64, 64, 64, 96, 'avg', 32),
            inc3b=F.InceptionBN(256, 64, 64, 96, 64, 96, 'avg', 64),
            inc3c=F.InceptionBN(320, 0, 128, 160, 64, 96, 'max', stride=2),
            inc4a=F.InceptionBN(576, 224, 64, 96, 96, 128, 'avg', 128),
            inc4b=F.InceptionBN(576, 192, 96, 128, 96, 128, 'avg', 128),
            inc4c=F.InceptionBN(576, 128, 128, 160, 128, 160, 'avg', 128),
            inc4d=F.InceptionBN(576, 64, 128, 192, 160, 192, 'avg', 128),
            inc4e=F.InceptionBN(576, 0, 128, 192, 192, 256, 'max', stride=2),
            inc5a=F.InceptionBN(1024, 352, 192, 320, 160, 224, 'avg', 128),
            inc5b=F.InceptionBN(1024, 352, 192, 320, 192, 224, 'max', 128),
            out_tag=F.Linear(1024 + 8, 3000),

            conva=F.Convolution2D(576, 128, 1, nobias=True),
            norma=F.BatchNormalization(128),
            lina=F.Linear(2048, 1024, nobias=True),
            norma2=F.BatchNormalization(1024),
            out_a_tag=F.Linear(1024 + 8, 3000),

            convb=F.Convolution2D(576, 128, 1, nobias=True),
            normb=F.BatchNormalization(128),
            linb=F.Linear(2048, 1024, nobias=True),
            normb2=F.BatchNormalization(1024),
            out_b_tag=F.Linear(1024 + 8, 3000),
        )

    def forward(self, x, z, train=True):
        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x))), 3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h))), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = F.relu(self.norma(self.conva(a)))
        a = F.relu(self.norma2(self.lina(a)))
        a = F.concat((a, z))

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = F.relu(self.normb(self.convb(b)))
        b = F.relu(self.normb2(self.linb(b)))
        b = F.concat((b, z))

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = F.reshape(h, (h.data.shape[0], 1024))
        h = F.concat((h, z))

        return a, b, h

    def tag(self, x, z, train=True):
        a, b, h = self.forward(x, z, train=train)
        tag_a = F.sigmoid(self.out_a_tag(a))
        tag_b = F.sigmoid(self.out_b_tag(b))
        tag = F.sigmoid(self.out_tag(h))
        tag = tag * 0.8 + tag_a * 0.1 + tag_b * 0.1
        return tag
