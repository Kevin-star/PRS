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



def find_reverse_set(G:nx.Graph, root:int, rumor: List[int]):
    level_nodes = {root: 0}
    current_level = [root]
    level = 1
    while current_level:
        next_level = []
        for node in current_level:
            resuccessors = set(G.predecessors(node)) - set(level_nodes.keys())
            level_nodes.update({s: level for s in resuccessors})
            next_level.extend(resuccessors)
        current_level = next_level
        if set(current_level)&set(rumor):
            break
        level += 1
    return list(set(level_nodes.keys())-set(rumor))
    # return list(set(level_nodes.keys())-(set(rumor)&set(level_nodes.keys())))




def generate_randomreverse_set(G:nx.Graph, root:int, rumor: List[int]):
    np.random.seed(42)
    popo =[0.1,0.05,0.01]
    level_nodes = {root: 0}
    current_level = [root]
    level = 1
    while current_level:
        next_level = []
        for node in current_level:
            resuccessors = set(G.predecessors(node)) - set(level_nodes.keys())
            for nodecur in resuccessors:
                if np.random.uniform(0, 1) < 1/G.in_degree(node):
                   level_nodes[nodecur]=level
                   next_level.append(nodecur)
        current_level = next_level
        if set(current_level)&set(rumor):
            break
        level += 1
    return list(level_nodes.keys())


def Calculate_lambda(G:nx.Graph, rumor: List[int], K:int):
    n = G.number_of_nodes()
    r1 = len(rumor)
    l = 1 
    epsilon = 0.5
    alpha = math.sqrt(math.log( n - r1) + math.log(2))
    
    log_part = math.log(math.comb(n - r1, K)) + l * math.log(n - r1) + math.log(2)
    beta = math.sqrt((1 - 1 / math.e) * log_part)
    
    numerator = 2 * (n - r1) * ((1 - 1 / math.e) * alpha + beta) ** 2
    denominator = epsilon ** 2
    
    lambda_value = numerator/denominator
    
    return lambda_value

def Calculate_LB(G:nx.Graph, rumor: List[int], K:int):
    p=0.1
    S = set()
    Score = {}
    for node in rumor:
    # 添加所有与R中的节点相邻且不在R中的节点
        neighbors = set(G.neighbors(node))
        S= neighbors - set(rumor)
    for v in S:
        tmp = 1
        presuccessors = set(G.predecessors(v))
        for u in presuccessors:
            if u in rumor:
                tmp = tmp*(1-1/G.in_degree(v))
        Score[v]=1-tmp
    sorted_dict = sorted(Score.items(), key=lambda item: item[1], reverse=True)[:K]
    LB = sum(value for _, value in sorted_dict)
    return sorted_dict, LB


def count_elements_in_sets(sets):
    # 将所有集合中的元素合并成一个列表
    all_elements = [element for s in sets for element in s]
    
    # 使用 Counter 统计每个元素出现的次数
    element_counts = Counter(all_elements)
    
    # 对结果按出现次数从高到低排序
    sorted_counts = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_counts

datasets = ['MathOverflow']
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
                
                Seedset=[]
                Upperbound = []
                Upperbound1 = []
                rumor_nodes = []
                # num_nodes=10
                pred_graph = read_graph_from_edgefile(f'data/SEALDataset/{dataset}/T60_pred_edge1.pt')
                             
                core_numbers = nx.core_number(pred_graph)

                tmp = sorted(core_numbers.items(), key=lambda item: item[1], reverse=True)
                i=0
                while len(rumor_nodes)<KR:
                    rumor_nodes.append(tmp[i][0])
                    i=i+1
                subgraphs = generate_snapshots(pred_graph, R, 42)
                [Rumorspread, infectnodes] = forward_influence_sketch(subgraphs,rumor_nodes)
                
                # start_time = time.time()       
                # ReverseSet = []
                # OSeedset = []
                # for w in range(0,R):
                #     for node in infectnodes[w]: 
                #         ReverseSet.append(find_reverse_set(subgraphs[w],node,rumor_nodes))                    
                # for w in range(0,K):
                #     counter = Counter()
                #     for collection in ReverseSet:
                #         counter.update(collection)
                #     Seednode=counter.most_common()[0][0]
                #     OSeedset.append(Seednode)
                #     ReverseSet = [a for a in  ReverseSet if Seednode not in a] 
                # time0 = time.time() - start_time 
                # # MonteOPBRR=compute_montesimu_spread(pred_graph, rumor_nodes, OSeedset)
                
                
                # start_time = time.time()       
                # ReverseSet = []
                # for w in range(0,R):
                #     for node in infectnodes[w]: 
                #         ReverseSet.append(find_reverse_set(subgraphs[w],node,rumor_nodes))    
 
                # Q = count_elements_in_sets(ReverseSet)
                # tmpdic = dict(Q)
                # Seednode=Q[0][0]
                # Seedset.append(Seednode)
                # Q = Q[1:]
                # for _ in range(K-1): 
                    
                #     check, node_lookup = False, 0
                    
                #     alreadyvisited = []
                    
                #     while not check:

                #         node_lookup += 1

                #         current = Q[0][0]
                        
                #         if current not in alreadyvisited:
                #             tmp = tmpdic[Q[0][0]]
                #             for item in ReverseSet[:]:
                                
                #                 if current in item:
                #                     if (set(item)&set(Seedset)):
                #                         tmp -= 1
                #             my_list = list( Q[0])
                #             my_list[1] = tmp
                #             Q[0] = tuple(my_list)

                #         Q = sorted(Q, key = lambda x: x[1], reverse = True)

                #         alreadyvisited.append(current)

                #         check = (Q[0][0] == current)
                    
                #     Seedset.append(Q[0][0])
                #     Q = Q[1:]     
  
                # time1 = time.time() - start_time 

                # MontePBRR=compute_montesimu_spread(pred_graph, rumor_nodes, Seedset)
                # MBrumor=forward_influence_sketchMCIC(subgraphs,rumor_nodes, Seedset)[0]
                
                
                start_time = time.time()  
                LB = Calculate_LB(pred_graph, rumor_nodes, K)[1]
                lambda1 = Calculate_lambda(pred_graph, rumor_nodes, K)
                N = math.ceil(lambda1/LB)
                ReverseSet = []
                TSeedset = []
                nodes = list(pred_graph.nodes())  # 获取图中所有节点的列表
                nodes = list(set(nodes) - set(rumor_nodes))
                for _ in range(N): 
                    v = random.choice(nodes)  # 随机选择一个节点
                    ReverseSet.append(generate_randomreverse_set(pred_graph,v,rumor_nodes)) 
                     
                ReverseSet = [a for a in  ReverseSet if set(a) & set(rumor_nodes)] 
                ReverseSet = [list(set(a) - set(rumor_nodes)) for a in ReverseSet] 
                # for item in ReverseSet[:]:    
                #     if set(rumor_nodes).isdisjoint(set(item)):
                #         ReverseSet.remove(item)
                
                for w in range(0,K):
                    counter = Counter()
                    for collection in ReverseSet:
                        counter.update(collection)
                    Seednode=counter.most_common()[0][0]
                    TSeedset.append(Seednode)
                    ReverseSet = [a for a in  ReverseSet if Seednode not in a] 
                time6 = time.time() - start_time 
                # TMontePBRR=compute_montesimu_spread(pred_graph, rumor_nodes, TSeedset)
                # TMBrumor=forward_influence_sketchMCIC(subgraphs,rumor_nodes, TSeedset)[0]
    
                
                # ######随机方法
                # start_time  = time.time()
                # RandomSeedset = []
                # while len(RandomSeedset)<K:
                #     tmpnode = random.sample(pred_graph.nodes(), 1)
                #     if tmpnode[0] not in rumor_nodes:
                #         RandomSeedset.append(tmpnode[0])
                # time2 = time.time() - start_time   
                # RandomBrumor=forward_influence_sketchMCIC(subgraphs, rumor_nodes, RandomSeedset)[0]
                # RandomRC=Rumorspread-RandomBrumor
                # MonteRandom = compute_montesimu_spread(pred_graph, rumor_nodes, RandomSeedset)        

                # #######最大出度方法
                # start_time  = time.time()
                # Maxdseedset = []
                # out_degree = list(pred_graph.out_degree)
                # tmpQ = sorted(out_degree, key = lambda x: x[1], reverse = True)
                # i=0
                # while len(Maxdseedset)<K:
                #     if tmpQ[i][0] not in rumor_nodes:
                #         Maxdseedset.append(tmpQ[i][0])
                #     i=i+1
                # time3 = time.time() - start_time  
                # MaxdBrumor=forward_influence_sketchMCIC(subgraphs, rumor_nodes, Maxdseedset)[0]
                # RandomRC=Rumorspread-MaxdBrumor
                # MonteMaxd = compute_montesimu_spread(pred_graph, rumor_nodes, Maxdseedset)        

                # #######最大Closeness centrality方法
                # start_time  = time.time()
                # MaxCSeedset = []
                # closeness_centrality = nx.closeness_centrality(pred_graph)
                
                # tmpCC = sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)
                # i=0
                # while len(MaxCSeedset)<K:
                #     if tmpCC[i] not in rumor_nodes:
                #         MaxCSeedset.append(tmpCC[i])
                #     i=i+1
                # time4 = time.time() - start_time  
                # MaxCCBrumor=forward_influence_sketchMCIC(subgraphs, rumor_nodes, MaxCSeedset)[0]
                # MaxBCRC=Rumorspread-MaxBCBrumor
                # MonteCC = compute_montesimu_spread(pred_graph, rumor_nodes, MaxCSeedset)   
                #######最大K-core方法
                # start_time  = time.time()
                # MaxKSeedset = []
                # core_numbers = nx.core_number(pred_graph)

                # tmpkcore = sorted(core_numbers.items(), key=lambda item: item[1], reverse=True)
                # i=0
                # while len(MaxKSeedset)<K:
                #     if tmpkcore[i][0] not in rumor_nodes:
                #         MaxKSeedset.append(tmpkcore[i][0])
                #     i=i+1
                # time5 = time.time() - start_time  
                # MaxKcoreBrumor=forward_influence_sketchMCIC(subgraphs, rumor_nodes, MaxKSeedset)[0]
                # MaxBCRC=Rumorspread-MaxBCBrumor
                # MonteKcore = compute_montesimu_spread(pred_graph, rumor_nodes, MaxKSeedset)   
                
                
                # data={'Time0':time0, 'Time1':time1, 'Time2':time2, 'Time3':time3, 'Time4':time4, 'Time5':time5, 'Time6':time6, 'MBOrumor':MonteOPBRR, 'MBrumor':MontePBRR, 'TMBrumor': TMontePBRR, 'RandomBrumor': MonteRandom, 'MaxdBrumor':MonteMaxd, 'MaxCCBrumor': MonteCC, 'MaxKcoreBrumor': MonteKcore}
                # # data={'Time1':time1,'MRS':MBrumor}
                # df = pd.DataFrame(data, index=[0])    
                # df.to_csv(f'WCRanDATA1{dataset}K{K}R{KR}SG{R}.csv', index=False)    
