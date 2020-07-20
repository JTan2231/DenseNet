import random
import time
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from net import DenseNet

BATCH_SIZE = 64
EPOCHS = 300
TOL = 0.002
TOL_DIFF = 0.01
LIMIT = 30
WEIGHTS = "/root/python/tensorflow/projects/repo/densenet/weights/final.h5"
WEIGHTS_COPY_FINAL = "/home/joey/python/tensorflow/projects/densenet/weights_copy_FINAL.h5"

# TODO: clean this mess
# TODO: actually train the network

def preprocess(data):
    data['image'] = tf.image.convert_image_dtype(data['image'], tf.float32)
    data['image'] = tf.pad(data['image'], [[4, 4], [4, 4], [0, 0]])
    data['image'] = tf.image.random_crop(data['image'], [32, 32, 3])
    if bool(random.getrandbits(1)):
        data['image'] = tf.image.flip_left_right(data['image'])

    return data

def trainEpoch(train_data,
               test_data,
               train_acc,
               test_acc,
               history,
               epoch,
               count,
               lr,
               net,
               opt,
               loss):
    epoch_start = time.time()
    i = 1.0
    for train in train_data:
        start = time.time()
        with tf.GradientTape() as tape:
            logits = net(train['image'], training=True)
            train_loss = loss(train['label'], logits)

        gradients = tape.gradient(train_loss, net.trainable_variables)
        opt.apply_gradients(zip(gradients, net.trainable_variables))

        train_accuracy(train['label'], logits)

        end = time.time()

        if i/(50000//BATCH_SIZE) >= 1:
            print("Training complete.")
        else:
            print("Training progress: {:.03f}, batch time: {:.03f} seconds, accuracy: {:.03f}\r".format(i/(50000//BATCH_SIZE), end-start, train_accuracy.result()), end="")

        i+=1

    i = 1.0
    for test in test_data:
        start = time.time()
        logits = net(test['image'])
        test_loss = loss(test['label'], logits)

        test_accuracy.update_state(test['label'], logits)
        
        end = time.time()

        if i/(10000//BATCH_SIZE) >= 1:
            print("Testing complete.\r", end="")
        else:
            print("Testing progress: {:.03f}, batch time: {:.03f} seconds, accuracy: {:.03f}\r".format(i/(10000//BATCH_SIZE), end-start, test_accuracy.result()), end="")
        i+=1

    epoch_end = time.time()
    print("Total epoch time: {:.03f}".format(epoch_end - epoch_start))

    if abs(test_accuracy.result() - history) < TOL:
        count += 1
        print("Count increased to {:01d}. Test accuracy (current, previous): {:.03f}, {:.03f}".format(count, test_accuracy.result(), history))

    elif (abs(test_accuracy.result() - history) > TOL_DIFF and count > 0):
        count = 0
        print("Count reset")

    history = test_accuracy.result()
    print("Epoch {:01d} Accuracies:\n\tTrain: {:.3f}\n\tTest: {:.3f}".format(epoch, train_accuracy.result(), test_accuracy.result()))

    return

history = 0

dataset = tfds.load('cifar10', shuffle_files=True)

train = dataset['train'].map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

#net = create_net(k=12, depth=40, theta=1, bottleneck=False)
net = DenseNet(k=12, depth=100, theta=0.5, bottleneck=True)

net.build_graph((None, 32, 32, 3))
net.summary()

#net.load_weights(WEIGHTS_COPY_FINAL)
#print("Weights loaded successfully")

lr = 0.0009
opt = tf.keras.optimizers.Adam(amsgrad=True)
#opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# training/validation loop
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
epoch = 1
count = 0

#print(net.trainable_variables)

for epoch in range(EPOCHS):
    train_accuracy.reset_states()
    test_accuracy.reset_states()
    # training loop
    trainEpoch(train, test, train_accuracy, test_accuracy,
               history, epoch, count, lr, net, opt, loss)
    net.save_weights(WEIGHTS)
