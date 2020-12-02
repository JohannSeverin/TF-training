import numpy as np
import os, sqlite3
from tqdm import tqdm

from pandas import read_sql
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph as knn

from spektral.data import Dataset, Graph

class IC_graphs(Dataset):
    """
    A class to convert the db files to a graph
    """
    def __init__(self, n_data = None, **kwargs):
       
        # Call super class 
        self.n_data = n_data
        
        super().__init__(**kwargs)


    def read(self):
    # We must return a list of Graph objects
        output = []

        self.n_data = len(os.listdir(os.path.join(self.path)))

        print("Loading data")
        for i in tqdm(range(self.n_data)):
            data = np.load(os.path.join(self.path, f'graph_{i}.npz'), allow_pickle=True)
            
            output.append(
                Graph(x=data['x'], a=data['a'], e = data['e'], y=data['y'])
            )
        print("Data loaded")

        return output


    def download(self, db_path = "raw_files/139008_00.db", n_data = None, k_neighbors = 6):
        # Make a folder for data
        print("Creating folder")
        os.mkdir(self.path)

        # Convert database to graphs
        if self.n_data == None:
            self.n_data = 99999999

        # Connect to database
        print("Downloading data")
        with sqlite3.connect(db_path) as con:
            seq     = read_sql(f"select * from sequential where event_no < {self.n_data};", con)
            sca     = read_sql(f"select * from scalar where event_no < {self.n_data};", con)
      

        # Check that both scalars and sequentials are availible 
        node_cols = ["dom_x", "dom_y", "dom_z", "dom_charge", "dom_time"]
        event_no = seq.event_no.unique()[sca.event_no.isin(sca.event_no)]

        print("Making Graphs")
        for i in tqdm(range(len(event_no))):
            id      = event_no[i]
            x       = np.array(seq.loc[seq.event_no == id, node_cols])
            pos     = x[:, :3]
            time    = x[:,  4]

            A       = knn(pos, k_neighbors)


            node_cols = ["dom_x", "dom_y", "dom_z", "dom_charge", "dom_time"]

            x       = np.array(seq.loc[seq.event_no == id, node_cols])
            pos     = x[:, :3]
            time    = x[:,  4]

            a       = knn(pos, k_neighbors)

            send    = np.repeat(np.arange(A.shape[0]), k_neighbors)
            receive = A.indices

            dists = np.linalg.norm(pos[receive] - pos[send], axis = 1)
            vects = normalize(pos[receive] - pos[send])
            dts   = time[receive] - time[send]

            e = np.vstack([dists, dts, vects.T])

            y = np.array(sca.loc[sca.event_no == 1])[:, 1:].flatten()

            filename = os.path.join(self.path, f'graph_{i}')
            np.savez(filename, x=x, a=a, e = e, y=y)

        self.data_n = len(event_no)







if __name__ == "__main__":
    os.system("rm -rf /home/johann/.spektral/datasets/IC_graphs")
    X = IC_graphs(10000)







