import networkx as nx
import dgl,random,torch,hashlib
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import torch.nn as nn


# PGNN RPE

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict

def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    dists_dict=single_source_shortest_path_length_range(graph, nodes, cutoff)
    return dists_dict

def precompute_dist_data(edge_index, num_nodes, approximate=0):
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist!=-1: dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array

def get_random_anchorset(n,m=0,c=0.5):
    m = int(np.log2(n)) if m==0 else m
    m = 1 if m<1 else m
    copy = int(c*m)
    copy = 1 if copy<1 else copy
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        anchor_size=1 if anchor_size<1 else anchor_size
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id

def get_dist_max(anchorset_id, dist):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id)))
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long()
    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        dist_argmax[:,i] = dist_argmax_temp
    return dist_max, dist_argmax

def preselect_anchor(num_nodes, dists, m, c):
    anchorset_id = get_random_anchorset(num_nodes,m,c)
    dists_max, dists_argmax = get_dist_max(anchorset_id, dists)
    return dists_max,dists_argmax

def get_dm_da(g,m=0,c=0.2,approx=0):
    eg=g.edges()
    eg=np.array([eg[0].numpy(),eg[1].numpy()])
    vg=list(g.nodes())
    dists = precompute_dist_data(eg, len(vg), approximate=approx)
    dists = torch.from_numpy(dists).float()
    return preselect_anchor(len(vg),dists,m,c)


# OTHER
    
class MethodWLNodeColoring:

    def setting_init(self, node_list, link_list):
        for node in node_list:
            self.node_color_dict[node] = 1
            self.node_neighbor_dict[node] = {}

        for pair in link_list:
            u1, u2 = pair
            if u1 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u1] = {}
            if u2 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u2] = {}
            self.node_neighbor_dict[u1][u2] = 1
            self.node_neighbor_dict[u2][u1] = 1

    def WL_recursion(self, node_list):
        iteration_count = 1
        while True:
            new_color_dict = {}
            for node in node_list:
                neighbors = self.node_neighbor_dict[node]
                neighbor_color_list = [self.node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(self.node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if self.node_color_dict == new_color_dict or iteration_count == self.max_iter:
                return
            else:
                self.node_color_dict = new_color_dict
            iteration_count += 1

    def run(self,node_list,link_list):
        self.max_iter = 2
        self.node_color_dict = {}
        self.node_neighbor_dict = {}
        self.setting_init(node_list, link_list)
        self.WL_recursion(node_list)
        return self.node_color_dict

def MethodGraphBatching(S,index_id_dict,k=7):
    user_top_k_neighbor_intimacy_dict = {}
    for node_index in index_id_dict:
        node_id = index_id_dict[node_index]
        s = S[node_index]
        s[node_index] = -1000.0
        top_k_neighbor_index = s.argsort()[-k:][::-1]
        user_top_k_neighbor_intimacy_dict[node_id] = []
        for neighbor_index in top_k_neighbor_index:
            neighbor_id = index_id_dict[neighbor_index]
            user_top_k_neighbor_intimacy_dict[node_id].append((neighbor_id, s[neighbor_index]))
    return user_top_k_neighbor_intimacy_dict

def MethodHopDistance(node_list,link_list,batch_dict):
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(link_list)

    hop_dict = {}
    for node in batch_dict:
        if node not in hop_dict: hop_dict[node] = {}
        for neighbor, score in batch_dict[node]:
            try:
                hop = nx.shortest_path_length(G, source=node, target=neighbor)
            except:
                hop = 99
            hop_dict[node][neighbor] = hop
    return hop_dict

def adj_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5, where=(rowsum!=0)).flatten()
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def cal_s(adj, c=0.15):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    eigen_adj = c * inv((sp.eye(adj.shape[0]) - (1 - c) * adj_normalize(adj)).toarray())
    return eigen_adj

def get_pes(dg):
    MWL=MethodWLNodeColoring()
    g=dg.to_networkx()
    eg=list(g.edges())
    vg=list(g.nodes())
    adj=nx.adjacency_matrix(g)
    S=cal_s(adj)
    ind={}
    for i in range(len(vg)): ind[i]=vg[i]
    
    wl_dict=MWL.run(np.array(vg),np.array(eg))
    batch_dict=MethodGraphBatching(S,ind)
    hop_dict=MethodHopDistance(vg,eg,batch_dict)   
        
    role_ids_list = []
    position_ids_list = []
    hop_ids_list = []
    for node in ind:
        node_index = ind[node]
        neighbors_list = batch_dict[node]
    
        role_ids = [wl_dict[node]]
        position_ids = range(len(neighbors_list) + 1)
        hop_ids = [0]
        for neighbor, intimacy_score in neighbors_list:
            neighbor_index = ind[neighbor]
            role_ids.append(wl_dict[neighbor])
            if neighbor in hop_dict[node]:
                hop_ids.append(hop_dict[node][neighbor])
            else:
                hop_ids.append(99)
        role_ids_list.append(role_ids)
        position_ids_list.append(position_ids)
        hop_ids_list.append(hop_ids)
    wl_embedding = torch.LongTensor(role_ids_list)
    hop_embeddings = torch.LongTensor(hop_ids_list)
    int_embeddings = torch.LongTensor(position_ids_list)
    return wl_embedding,hop_embeddings,int_embeddings


if __name__=='__main__':
    # PGNN
    # from model import PGNN_layer
    
    dg = dgl.DGLGraph()
    dg.add_nodes(7)    
    for i in [[0,1],[0,3],[3,5],[1,2],[1,4],[3,6]]:
        dg.add_edges(i[0],i[1])
        dg.add_edges(i[1],i[0])
    
    # dm,da=get_dm_da(dg)
    # h=torch.randn(7,512)
    
    # M=PGNN_layer(512,512,dist_trainable=False)
    # q,w=M(h,dm,da)
    
    # OTHER
    WLE,HOP,INT=get_pes(dg)

    wl_role_embeddings = nn.Embedding(100, 512)
    inti_pos_embeddings = nn.Embedding(100, 512)
    hop_dis_embeddings = nn.Embedding(100, 512)
    
    role_embeddings = torch.sum(wl_role_embeddings(WLE),1)
    position_embeddings = torch.sum(inti_pos_embeddings(HOP),1)
    hop_embeddings = torch.sum(hop_dis_embeddings(INT),1)
    
    