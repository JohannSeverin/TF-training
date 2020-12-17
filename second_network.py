import os, pickle, inspect
from tqdm import tqdm

import tensorflow as tf
import numpy as np

from spektral.layers import MessagePassing, GlobalSumPool, ECCConv
from spektral.data.loaders import DisjointLoader

from sonnet.nets import MLP
from sonnet import Linear

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# ==========================================================================
# Set variables
# ==========================================================================
learning_rate = 1e-3
batch_size    = 64
epochs        = 3
hidden_states = 30
layers        = 2
test_size     = 0.1


# ==========================================================================
# Message-passing model
# ==========================================================================
# class GNN(MessagePassing):
#     def __init__(self, n_out, activation):
#         super().__init__(activation = activation)
#         self.n_out = n_out
       
#     def build(self, input_shape):
#         self.n_in = input_shape
#         self.msg_mlp = MLP([hidden_states] * layers)
#         self.upd_mlp = MLP([hidden_states, 5])


#     def message(self, x, e):
#         x = tf.concat([self.get_i(x), self.get_j(x), e], axis = 0)
#         output = self.msg_mlp(x)
#         return output
    
#     def update(self, x):
#         self.output  = self.upd_mlp(x)
#         return self.upd_mlp

#     def call(self, inputs, **kwargs):
#         x, a, e = self.get_inputs(inputs)
#         return self.propagate(x, a, e)
    
    
#     def propagate(self, x, a, e=None, **kwargs):
#         self.n_nodes = tf.shape(x)[0]
#         self.index_i = a.indices[:, 1]
#         self.index_j = a.indices[:, 0]

#         # Message
#         msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs) # Get kwargs not working. 
#         messages = self.message(x, e) # Just add e manually

#         # Aggregate
#         agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)
#         embeddings = self.aggregate(messages, **agg_kwargs)

#         # Update
#         upd_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)
#         output = self.update(embeddings, **upd_kwargs)

#         return output




# ============================================================================
# Load Data
# ============================================================================
print("Loading data")
from graph_gen import IC_graphs
data = IC_graphs()


# Parameters
F = data.n_node_features
S = data.n_edge_features
n_out = data.n_labels


# Train_test
print("Splitting data")
idxs  = np.random.permutation(len(data))
split = int((1 - test_size) * len(data))
idx_tr, idx_test  = np.split(idxs, [split])
dataset_train, dataset_test = data[idx_tr], data[idx_test]

# Creating loaders
print("Creating loaders")
loader_train = DisjointLoader(dataset_train, batch_size = batch_size, epochs = epochs)
loader_test  = DisjointLoader(dataset_test, batch_size = batch_size, epochs = epochs)


# ============================================================================
# Build Model
# ============================================================================
X_in = Input(shape=(F,), name='X_in')
A_in = Input(shape=(None,), sparse=True, name='A_in')
E_in = Input(shape=(S,), name='E_in')
I_in = Input(shape=(), name='segment_ids_in', dtype=tf.int64)

X_1  = ECCConv(32, [hidden_states, hidden_states * 4, hidden_states * 4, hidden_states], activation = "relu", n_out = hidden_states)([X_in, A_in, E_in])
X_2  = GlobalSumPool()([X_1, I_in])
output = Dense(n_out)(X_2)

# Build model
model = Model(inputs =  [X_in, A_in, E_in, I_in], outputs = output)
opt   = Adam(lr = learning_rate)
loss_fn = MeanSquaredError()


# ==========================================================================
# Fit Model
# ==========================================================================
@tf.function(input_signature = loader_train.tf_signature(), experimental_relax_shapes = True)
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training = True)
        loss        = loss_fn(targets, predictions)
        loss        += sum(model.losses) 
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


print("Fitting model")
current_batch = 0
model_loss    = 0

for batch in loader_train:
    outs = train_step(batch[0], np.expand_dims(batch[1], -1))
    current_batch += 1
    print(f"completed: \t {current_batch} \t / {loader_train.steps_per_epoch} \t current_loss: {model_loss / current_batch}", end ='\r' )
    model_loss += outs
    if current_batch == loader_train.steps_per_epoch:
        print('Loss: {}'.format(model_loss / loader_train.steps_per_epoch))
        model_loss = 0
        current_batch = 0

    

