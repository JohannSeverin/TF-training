import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

# Define model class
from tensorflow.keras import Model # Parent class



import tensorflow_probability as tfp
tfd = tfp.distributions
# Libary for probability distributions in tensorflow

class gauss_reg(Model):
    def __init__(self, guesses = (1., 1.)):
        super(gauss_reg, self).__init__()
        self.mean = tf.Variable(guesses[0])
        self.sigma = tf.Variable(guesses[1])

    def call(self, inputs):
        return tfd.Normal(self.mean, self.sigma).prob(inputs)



# Data for training 
data_gauss = tf.random.normal([1000], mean = 30.5, stddev = 15.)

def loss(model, inputs):
    likelihoods = tf.math.log(model(inputs))
    log_like = - 2 * tf.reduce_sum(likelihoods)
    return log_like

def grad(model, inputs):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs)
    return tape.gradient(loss_value, [model.mean, model.sigma])



model = gauss_reg(guesses = (10., 10.))
optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-3, momentum = 0.1)

print("Initial loss: {:.3f}".format(loss(model, data_gauss)))

steps = 1000

for i in range(steps):
    grads = grad(model, data_gauss)
    optimizer.apply_gradients(zip(grads, [model.mean, model.sigma]))
    if i % 100 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, data_gauss)))
        print(f"Estimate: mean = {model.mean.numpy():.5f}, sigma = {model.sigma.numpy():.5f}")
        print("\n")



print(f"Final loss: {loss(model, data_gauss):.3f}")



print(f"Estimate: mean = {model.mean.numpy():.5f}, sigma = {model.sigma.numpy():.5f}")





