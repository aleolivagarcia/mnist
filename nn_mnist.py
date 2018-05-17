import gzip
import _pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = _pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

#Defining NN

y_data = one_hot(train_y.astype(int), 10)
y_valid = one_hot(valid_y.astype(int), 10)
y_test = one_hot(test_y.astype(int), 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 100)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(100)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(100, 20)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)

W3 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)
b3 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
j = tf.nn.sigmoid(tf.matmul(h, W2) + b2)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(j, W3) + b3)

loss = tf.reduce_sum(tf.square(y_ - y))

lrate=0.05
train = tf.train.GradientDescentOptimizer(lrate).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("Parámetros:\nCapas ocultas: 2\nNúmero de neuronas en la capa 1: 100\n"
      "Número de neuronas en la capa 2: 20\nLearning Rate:", lrate)

batch_size = 20

error=9999999
epoch = 1
errorArray = []
while True:
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    result = sess.run(y, feed_dict={x: batch_xs})

    errorPrev=error
    error = sess.run(loss, feed_dict={x: valid_x, y_: y_valid})
    errorArray.append(error)

    if (abs(error-errorPrev)/errorPrev) < 0.001 or error > errorPrev:
        break
    epoch += 1

print ("----------------------------------------")

test_result = sess.run(y, feed_dict={x: test_x})
aciertos=0
for nn, real in zip(y_test, test_result):
    if np.argmax(nn) == np.argmax(real):
        aciertos += 1

precision =  aciertos/len(y_test)*100
print("Precisión de la red:", precision, "%")

plt.subplot(1, 2, 1)
plt.plot(errorArray)
plt.xlabel("Epochs")
plt.ylabel("Error de validación")
plt.title("Variación del error")
plt.show()
