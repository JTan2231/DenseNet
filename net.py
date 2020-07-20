import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

DEFAULT_PARAMS = { 'k': 12, 'depth': 100, 'theta': 0.5, 'bottleneck': True }

class DenseLayer(layers.Layer):
    def __init__(self, k, bottleneck=False):
        super(DenseLayer, self).__init__()

        self.necked = bottleneck

        if bottleneck:
            self.bn1 = layers.BatchNormalization()
            self.relu1 = layers.ReLU()
            self.bottleneck = layers.Conv2D(4*k, 1, padding='same')

        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.conv = layers.Conv2D(k, 3, padding='same')

        if not self.necked:
            self.dropout = layers.Dropout(rate=0.2)
        self.concat = layers.Concatenate()

    def call(self, input_tensor, training):
        conv = input_tensor

        if self.necked:
            conv = self.bn1(conv, training)
            conv = self.relu1(conv)
            conv = self.bottleneck(conv)

        conv = self.bn2(conv, training)
        conv = self.relu2(conv)
        conv = self.conv(conv)

        if not self.necked:
            conv = self.dropout(conv, training)
        return self.concat([input_tensor, conv])

    def build_graph(self, input_shape):
        no_batch = input_shape[1:]
        self.build(input_shape)

        inputs = keras.Input((no_batch))
        self.call(inputs)

class Transition(layers.Layer):
    def __init__(self, input_filters, theta=0.5):
        super(Transition, self).__init__()
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv = layers.Conv2D(int(input_filters*theta), 1, padding='same')
        self.pool = layers.AveragePooling2D(2, strides=2)

    def call(self, input_tensor, training):
        conv = self.bn(input_tensor, training)
        conv = self.relu(conv)
        conv = self.conv(conv)
        return self.pool(conv)

class DenseBlock(layers.Layer):
    def __init__(self, k, depth, bottleneck):
        super(DenseBlock, self).__init__()
        self.lays = []
        for i in range(depth):
            self.lays.append(DenseLayer(k, bottleneck))

    def call(self, input_tensor, training):
        conv = input_tensor
        for layer in self.lays:
            conv = layer(conv, training)

        return conv

class DenseNet(keras.Model):
    def __init__(self, k=DEFAULT_PARAMS['k'],
                       depth=DEFAULT_PARAMS['depth'],
                       theta=DEFAULT_PARAMS['theta'],
                       bottleneck=DEFAULT_PARAMS['bottleneck']):
        super(DenseNet, self).__init__()

        self.conv_init = layers.Conv2D(2*k, 3, padding='same')

        if bottleneck:
            block_depth = ((depth-3) // 3) // 2
        else:
            block_depth = (depth-1) // 3

        self.block1 = DenseBlock(k, block_depth, bottleneck)
        self.trans1 = Transition(2*k + k*block_depth, theta=theta)
        self.block2 = DenseBlock(k, block_depth, bottleneck)
        self.trans2 = Transition((2*k + k*block_depth)/2 + k*block_depth, theta=theta)
        self.block3 = DenseBlock(k, block_depth, bottleneck)

        self.pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(units=10, activation='softmax')
        
    def call(self, input_tensor, training=False):
        conv = self.conv_init(input_tensor)

        conv = self.block1(conv, training)
        conv = self.trans1(conv, training)
        conv = self.block2(conv, training)
        conv = self.trans2(conv, training)
        conv = self.block3(conv, training)

        conv = self.pool(conv)
        conv = self.flatten(conv)
        return self.dense(conv)

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)#_nobatch)

        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        self.__call__(inputs, True)
