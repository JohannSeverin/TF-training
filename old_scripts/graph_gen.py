import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph as knn
from tqdm import tqdm
import os 
import pickle

k_neighbors = 6
n_data = 1000000

name = "data1_graphs.dat"

seq = pd.read_pickle("csv_files/sequential.dat")
sca = pd.read_pickle("csv_files/scalar.dat")

node_cols = ["dom_x", "dom_y", "dom_z", "dom_charge", "dom_time"]
node_pos  = ["dom_x", "dom_y", "dom_z"]

event_no = seq.event_no.unique()[sca.event_no.isin(sca.event_no)]

Xs     = []
As     = []
target = []

for i in tqdm(event_no[:n_data]):
    node_features = np.array(seq.loc[seq.event_no == i, node_cols])
    N             = len(node_features)
    
    node_pos      = node_features[:, :3]
    
    A             = knn(node_pos, k_neighbors)
    
    scalar        = np.array(sca.loc[sca.event_no == i, :])
    
    Xs.append(node_features)
    As.append(A)
    target.append(scalar)
    

if "graph_lists" not in os.listdir():
    os.mkdir("graph_lists")

pickle.dump((Xs, As, target), open(f"graph_lists/{name}", "wb"))

