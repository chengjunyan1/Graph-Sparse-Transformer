"""Torch modules for graph attention networks(GAT). Modified to a Transformer MHA."""
# pylint: disable= no-member, arguments-differ, invalid-name
# src: https://docs.dgl.ai/en/0.4.x/_modules/dgl/nn/pytorch/conv/gatconv.html
import math
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

# pylint: enable=W0235
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 d_k, 
                 d_v,
                 num_heads,
                 attn_drop=0.,
                 diffuse=0,
                 alpha=0.15):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, _ = expand_as_pair(in_feats)
        self._d_k = d_k
        self._d_v = d_v
        
        self.fc_K = nn.Linear(self._in_src_feats, d_k * num_heads)
        self.fc_Q = nn.Linear(self._in_src_feats, d_k * num_heads)
        self.fc_V = nn.Linear(self._in_src_feats, d_v * num_heads)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.output = nn.Linear(num_heads * d_v, self._in_src_feats)
        self.diffuse=diffuse
        self.alpha=alpha
        

    def forward(self, graph, h_K, h_Q, h_V):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()
        feat_K = self.fc_K(h_K).view(-1, self._num_heads, self._d_k)
        feat_Q = self.fc_Q(h_Q).view(-1, self._num_heads, self._d_k) 
        feat_V = self.fc_V(h_V).view(-1, self._num_heads, self._d_v)
        feat_Q /= math.sqrt(self._d_v)
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        graph.srcdata.update({'ft': feat_V, 'el': feat_Q})
        graph.dstdata.update({'er': feat_K})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_mul_v('el', 'er', 'e'))
        e = th.sum(graph.edata.pop('e'), -1, keepdim=True)
        
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)/1.25) # may bug in DGL, softmax summing up to 1.25 not 1.0
        # message passing
        if self.diffuse!=0:
            z0=graph.srcdata['ft']
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            context_diffused=graph.dstdata['ft']
            for i in range(self.diffuse):
                context=(1-self.alpha)*context_diffused+self.alpha*z0
                graph.srcdata['ft']=context
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                context_diffused=graph.dstdata['ft']
            rst=context
        else:
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
        rst = th.reshape(rst,[-1, self._num_heads*self._d_v])
        return self.output(rst)
    


if __name__=='__main__':
    
    import dgl,torch
    from prettytable import PrettyTable

    def layer_wise_parameters(model):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in model.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table


    def buildDG(nodes,edges):
        node_map={}
        num_node=len(nodes)
        g = dgl.DGLGraph()
        g.add_nodes(num_node)   
        for i in range(num_node):
            node_map[nodes[i]]=i
        for e in edges:
            g.add_edge(node_map[e[0]],node_map[e[1]])
        return g

    def buildBG(nodes,edges):    
        gs=[]
        for i in range(len(nodes)):
            gs.append(buildDG(nodes[i],edges[i]))
        return dgl.batch(gs)

    embed_size=3
    seq_length=2
    num_heads=1
    num_block=2
    d_k=3
    d_v=3
    d_ff=2048 
    
    gnode=[[1,3,5,9],[1,7]]  
    gedge=[[[3,1],[5,1],[9,1]],[[1,7]]]
    bg=buildBG(gnode,gedge)
    h=th.randn(6,embed_size)
    nids=th.cat([g.nodes() for g in dgl.unbatch(bg)])
    
    gat=GATConv(embed_size, embed_size, d_k, d_v,num_heads,attn_drop=0.2,diffuse=2,alpha=[0.15])
    q=gat(bg,h,h,h)
    
    
