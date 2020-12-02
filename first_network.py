import numpy as np
import tensorflow as tf

from tqdm import tqdm

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.layers import GraphConv, GlobalAvgPool, ops
from spektral.utils import batch_iterator, numpy_to_disjoint

## SETUP P  ARAMETERS FOR TRAINING
learning_rate = 1e-3
epochs = 5
batch_size = 32

data_file = "graph_lists/data1_graphs.dat"


## LOAD DATA
print("Loading Data")
from pandas import read_pickle
X, A, y = read_pickle(data_file)
y = np.squeeze(np.array(y))   # remove accidental dimension
y = np.expand_dims(y[:, 1], 1)
print("Data_loaded")


# Read dimensions
X_dim = X[0].shape[-1]
y_dim = y[0].shape[-1]

# train_test_split
from sklearn.model_selection import train_test_split
A_train, A_test, \
X_train, X_test, \
y_train, y_test = train_test_split(A, X, y, test_size = 0.1, random_state = 0)

# define scaler
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(np.vstack(X_train))


## BUILD MODEL - note: I have no Idea what I'm doing

# input layers
X_in = Input(shape=(X_dim,), name = "X_in")                          # X input layer
A_in = Input(shape=(None,), sparse = True, name = "A_in")            # A input layer
I_in = Input(shape = (), name = "segment_ids_in", dtype = tf.int32)  # Graph of ID in disjoint mode

# layers, we are going to try with two conv layers and a global sum pool
X_1 = GraphConv(32, activation = "relu")([X_in, A_in])
X_2 = GraphConv(32, activation = "relu")([X_1, A_in])
X_3 = GlobalAvgPool()([X_2, I_in])
output = Dense(y_dim)(X_3)


# model:
model = Model(inputs=[X_in, A_in, I_in], outputs = output)
optimizer = Adam(lr = learning_rate)
loss_func = MeanSquaredError()

# Train step definition in tensorflow
@tf.function(
    input_signature = (tf.TensorSpec((None, X_dim), dtype = tf.float64), # X spec
                       tf.SparseTensorSpec((None, None), dtype = tf.float64), # A spec
                       tf.TensorSpec((None,), dtype = tf.int32), # I spec
                       tf.TensorSpec((None, y_dim))),
    experimental_relax_shapes = True)
def train_step(X_, A_, I_, y_):
    with tf.GradientTape() as tape:
        predictions = model([X_, A_, I_], training = True)
        loss = loss_func(y_, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss




### FIT the model
current_batch = 0
model_loss    = 0
batches_in_epoch = np.ceil(len(A_test) / batch_size)

print("Fitting model")
batches_train = batch_iterator([X_train, A_train, y_train], \
                                batch_size = batch_size, epochs = epochs)

for b in batches_train:
    X_, A_, I_ = numpy_to_disjoint(*b[:-1])
    X_ = scale.transform(X_)
    A_ = ops.sp_matrix_to_sp_tensor(A_)
    y_ = b[-1]
    outs = train_step(X_, A_, I_, y_)

    model_loss += outs.numpy()
    current_batch += 1
    if current_batch == batches_in_epoch:
        print('Loss: {}'.format(model_loss / batches_in_epoch))
        current_batch = 0
        model_loss = 0

# Evaluating
print("Testing model")

model_loss = 0
batches_in_epoch = np.ceil(len(A_test) / batch_size)
batches_test = batch_iterator([X_test, A_test, y_test], batch_size=batch_size)

for b in batches_test:
    X_, A_, I_ = numpy_to_disjoint(*b[:-1])
    X_ = scale.transform(X_)
    A_ = ops.sp_matrix_to_sp_tensor(A_)
    y_ = b[-1]

    predictions = model([X_, A_, I_], training = False)
    model_loss += loss_func(y_, predictions)

model_loss /= batches_in_epoch

print(f"Done, test loss:{model_loss}")