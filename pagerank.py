import networkx as nx
from networkx.exception import NetworkXError

import pandas as pd

import numpy as np
import scipy.sparse
import pickle


class PageRank():
    
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.load_data()

        self.G = nx.DiGraph()
        self.G.add_nodes_from([i for i in range(len(self.entity_list))])
        self.G.add_edges_from(self.edges)
        
    def load_data(self):
        self.triplet_df = pd.read_csv(self.data_dir +'triplet.csv')
        self.edges = [[r[0], r[1]] for r in self.triplet_df.values]

        self.user_list = []
        self.item_list = []
        self.entity_list = []
        with open(self.data_dir + 'user_list.txt', 'r') as f:
            for l in f:
                self.user_list.append(l.replace('\n', ''))
        with open(self.data_dir + 'item_list.txt', 'r') as f:
            for l in f:
                self.item_list.append(l.replace('\n', ''))
        with open(self.data_dir + 'entity_list.txt', 'r') as f:
            for l in f:
                self.entity_list.append(l.replace('\n', ''))