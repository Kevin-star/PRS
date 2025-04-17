# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:50:22 2024

@author: 28300
"""


import random
import json
import os.path as osp
from collections import defaultdict
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import sketch_based_greedy_RTlL, order_based_SBG_RTlL, build_upper_bound_label, calculate_candidate_edges, generate_user_groups
from utils import read_temporary_graph_data, read_graph_from_edgefile
from utils import draw_networkx, draw_evaluation
import time
import copy
import random
import heapq
from typing import List
from functools import reduce
import numpy as np
from tqdm import tqdm
import heapq
import networkx as nx
from collections import Counter
import math
from pathlib import Path

import networkx as nx
import pandas as pd

import torch



def read_temporary_graph_data(filename):
    """Read temporary graph data from file.

    Args:
        filename (str or Path): the temporary graph data file path.
        timespan (int): the time span of data in days.
        T (float): the number of temporary graphs segmented with timestamp.

    Returns:
        List[nx.DiGraph]: the list of temporary direted graphs.
    """
    
    if isinstance(filename, str):
        filename = Path(filename)
    if not filename.exists():
        raise ValueError(f"{filename} does not exist.")

    if filename.suffix == '.gz':
        data = pd.read_csv(filename, compression='gzip', sep=' ', names=['SRC', 'TGT', 'TS'])
    else:
        raise ValueError(f"{filename.suffix} is not supported to read.")


    graph = nx.from_pandas_edgelist(data, 'SRC', 'TGT', create_using=nx.DiGraph)
    
    # remove isolate node and self loop
    graph.remove_nodes_from(nx.isolates(graph))
    graph.remove_edges_from(nx.selfloop_edges(graph))
 
    
    return graph

def generate_snapshots(graph: nx.Graph, r: int, seed: int = 42):
    """Generate r random sketch graph by removing each edges with probability 1-P(u,v), which defined as 1/degree(v). 

    Args:
        graph (nx.Graph): graph object
        r (int): the number of snapshots generated
        seed (int): the random seed of numpy

    Returns:
        snapshots (List[nx.Graph]): r number sketch subgraph
    """

    np.random.seed(seed)
    
    popo =[0.1,0.05,0.01]
    snapshots = []
    for _ in range(r):
        # select_edges = [edge for edge in graph.edges if np.random.uniform(0, 1) < random.choice(popo)]
        select_edges = [edge for edge in graph.edges if np.random.uniform(0, 1) < 1/graph.degree(edge[1])]
        snapshots.append(graph.edge_subgraph(select_edges))

    return snapshots



def compute_montesimu_spread(graph: nx.Graph, Rumors: List[int], Seedsets: List[int]):
# def compute_montesimu_spread(graph: nx.Graph, users: List[int]):
    """Compute independent cascade in the graph

    Args:
        graph (nx.Graph): graph object
        users (List[int]): a set of user nodes
        mask (List[int], optional): a set of unparticipation cascade mask nodes. default is []. 

    Returns:
        spread, reached_node (int, List[int]): the number of vertexes reached by users in graph and the set of nodes reached by users.
    """
    spread=[]
    
    popo =[0.1,0.05,0.01]

    for i in range(10000):
        
        Rflag={}
        Cflag={}
        Rnew_active, Ractive = [], []
        Snew_active, Sactive = [], []
        Rnew_active, Ractive = Rumors[:], Rumors[:]
        Snew_active, Sactive = Seedsets[:],Seedsets[:]
        for node in graph:
            Rflag[node]=0
            Cflag[node]=0
        for node in Rnew_active:
            Rflag[node]=1
        for node in Snew_active:
            Cflag[node]=1   
        # for each newly activated nodes, find its neighbors that becomes activated
        np.random.seed(i)
        while Rnew_active:
            Sactivated_nodes = []
            for node in Snew_active:
                neighbors = list(graph.neighbors(node))
                for nodecur in neighbors:
                    if np.random.uniform(0,1) < 1/graph.degree(nodecur):
                        if Rflag[nodecur]==0:
                            Sactivated_nodes.append(nodecur)
                            Cflag[nodecur]=1
            
            # ensure the newly activated nodes doesn't already exist
            Snew_active = list(set(Sactivated_nodes) - set(Sactive))
            Sactive += Snew_active
            
            Ractivated_nodes = []
            for node in Rnew_active:
                neighbors = list(graph.neighbors(node))
                for nodecur in neighbors:
                    if np.random.uniform(0,1) < 1/graph.degree(nodecur):
                        if Cflag[nodecur]==0:
                            Ractivated_nodes.append(nodecur)
                            Rflag[nodecur]=1
            
            # ensure the newly activated nodes doesn't already exist
            Rnew_active = list(set(Ractivated_nodes) - set(Ractive))
            Ractive += Rnew_active 

        spread.append(len(Ractive))
    return np.mean(spread)

def forward_influence_sketch(graphs: List[nx.Graph], users: List[int]):
    """The Forward Influence Sketch method

    Args:
        graphs (List[nx.Graph]): a set of snapshot graph
        users (List[int]): a group of user nodes

    Returns:
        spread (float): the mean of additional spread of vertexes reached by users in all sketch subgraph.
    """
    spread = []
    infected_nodes=[]

    for graph in graphs:
        [tmp_spread, tmp_infected_nodes] = compute_independent_cascade(graph, users)
        tmp_infected_nodes = list(set(tmp_infected_nodes) - set(users))
        spread.append(tmp_spread)
        infected_nodes.append(tmp_infected_nodes)       
        
    return np.mean(spread), infected_nodes

def compute_independent_cascade(graph: nx.Graph, users: List[int], mask: List[int] = []):
    """Compute independent cascade in the graph

    Args:
        graph (nx.Graph): graph object
        users (List[int]): a set of user nodes
        mask (List[int], optional): a set of unparticipation cascade mask nodes. default is []. 

    Returns:
        spread, reached_node (int, List[int]): the number of vertexes reached by users in graph and the set of nodes reached by users.
    """
    
    new_active, active = users[:], users[:]
        
    # for each newly activated nodes, find its neighbors that becomes activated
    while new_active:
        activated_nodes = []
        for node in new_active:
            if graph.has_node(node):
                # determine neighbors that become infected
                neighbors = list(graph.neighbors(node))
                activated_nodes += neighbors
        
        # ensure the newly activated nodes doesn't already exist
        new_active = list(set(activated_nodes) - set(active) - set(mask))
        active += new_active

    return len(active), active







# datasets = ['WikiTalk']
datasets = ['AskUbuntu','StackOverflow','EmailEuCore','WikiTalk']
SeedsizeR = [10,20,30]  
Seedsize = [5,10,15,20,25,30]
FinalR = [300]  
# SeedsizeR = [30]  
# Seedsize = [20]  
# datasets = ['MathOverflow']
# R=550
P=0.1



for R in FinalR:
    for dataset in datasets:
        for KR in SeedsizeR:
            for K in Seedsize:   
                
               
               
                
                
                data={'Time0':time0, 'Time1':time1, 'Time2':time2, 'Time3':time3, 'Time4':time4, 'Time5':time5, 'Time6':time6, 'MBOrumor':MonteOPBRR, 'MBrumor':MontePBRR, 'TMBrumor': TMontePBRR, 'RandomBrumor': MonteRandom, 'MaxdBrumor':MonteMaxd, 'MaxCCBrumor': MonteCC, 'MaxKcoreBrumor': MonteKcore}
                # data={'Time1':time1,'MRS':MBrumor}
                df = pd.DataFrame(data, index=[0])    
                df.to_csv(f'WCMaxKcoreDATA1{dataset}K{K}R{KR}SG{R}.csv', index=False)    
