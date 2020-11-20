import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from itertools import cycle

#NODE_COUNT = 196591
#EDGE_COUNT = 950327
#CHECKIN_COUNT = 6442890
#EDGES_DATA_FILE = "gowalla/Gowalla_edges_15k.txt"
#CHECKINS_DATA_FILE = "gowalla/Gowalla_totalCheckins_15k.txt"

NODE_COUNT = 15000
EDGES_DATA_FILE = "gowalla/Gowalla_edges_15k.txt"
CHECKINS_DATA_FILE = "gowalla/Gowalla_totalCheckins_15k.txt"


def getGraph():
    edges = pd.read_csv(EDGES_DATA_FILE, delim_whitespace=True,
                        header=None, names=['A', 'B'])

    return sparse.csr_matrix(
        (np.ones(edges.shape[0], dtype=bool),
         (edges['A'], edges['B'])), dtype=bool
    )


def getCheckins():
    info = pd.read_csv(CHECKINS_DATA_FILE, delim_whitespace=True, header=None,
                       names=['UserID', 'Time', 'Latitude', 'Longitude', 'LocationID'])

    users = np.asarray(info['UserID'])
    locations = np.asarray(info['LocationID'])
    return users, locations