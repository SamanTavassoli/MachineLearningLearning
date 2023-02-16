import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

'''
Following new tutorial https://www.youtube.com/watch?v=wQ8BIBpya2k&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&ab_channel=sentdex
Because the old one is out of date

feed forward
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) > weights > output layer

backprop
compare output to intended output > cost or loss function (ex. cross entropy)
optimization function (optimizer) > minimize cost (ex. AdamOptimizer, SGD, AdaGrad...)

feed forward + backprop = epoch
'''

mnist = tf.keras.datasets.mnist  # 28x28 images of hand written digits 0-9

# plt.imshow(x_train[0], cmap='Greys')
# plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())  # input layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # hidden layer 1
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # hidden layer 2
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer (10 outputs, activ. is porb dist)

model.compile(optimizer='adam',  # just a start optimizer
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

validation_loss, validation_accuracy = model.evaluate(x_test, y_test)

print(validation_loss, validation_accuracy)


# saving and loading

# model.save('epic_num_reader.model')
# new_model = tf.keras.models.load_model('epic_num_reader.model')
# predictions = new_model.predict([x_test])
#
# print(np.argmax(predictions[0]))
# plt.imshow(x_test[0])
# plt.show()




