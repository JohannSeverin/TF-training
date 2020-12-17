import numpy as np
import os, sqlite3, pickle
from tqdm import tqdm
import os.path as osp

from pandas import read_sql
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph as knn

from spektral.data import Dataset, Graph

class IC_graphs(Dataset):
    """
    A class to convert the db files to a graph
    """
    def __init__(self, n_data = None, **kwargs):
        self.n_data = n_data
        super().__init__(**kwargs)


    def read(self):
    # We must return a list of Graph objects
        output = pickle.load(open(self.path + "/IC_graphs.dat", 'rb'))
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

        # Graph making loop
        print("Making Graphs")
        graph_list = []
        for i in tqdm(range(len(event_no))):
            # Take id from list and make array of X array
            id      = event_no[i]
            x       = np.array(seq.loc[seq.event_no == id, node_cols])
            pos     = x[:, :3]
            time    = x[:,  4]

            # Adjacency Matrix
            A       = knn(pos, k_neighbors)
            
            # Steup note attributes
            send    = np.repeat(np.arange(A.shape[0]), k_neighbors)
            receive = A.indices

            dists = np.linalg.norm(pos[receive] - pos[send], axis = 1)
            vects = normalize(pos[receive] - pos[send])
            dts   = time[receive] - time[send]

            e = np.vstack([dists, dts, vects.T]).T

            y = np.array(sca.loc[sca.event_no == id, :])
            y = np.array(y[0][1])

            # filename = os.path.join(self.path, f'graph_{i}')
            graph_list.append(Graph(x, A, e, y))
            # np.savez(filename, x=x, a=a, e = e, y=y)
        
        print("Saving")
        pickle.dump(graph_list, open(self.path + "/IC_graphs.dat", 'wb'))
        self.data_n = len(event_no)
    
    @property
    def path(self):
        return osp.expanduser("~/data/IC_graph")







if __name__ == "__main__":
    os.system("rm -rf ~/home/johann/data/IC_graphs")
    X = IC_graphs()







