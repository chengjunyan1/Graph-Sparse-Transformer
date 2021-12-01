"""
Implementation of "Attention is All You Need"
"""

import math, torch
import torch.nn as nn

from c2nl.modules.position_ffn import PositionwiseFeedForward
from c2nl.modules.util_class import LayerNorm
from c2nl.utils.misc import aeq


class MultiHeadedDiffusedAttention(nn.Module):

    def __init__(self, head_count, model_dim, d_k, d_v, dropout=0.1, diffuse=0, alpha=0.15):
        super(MultiHeadedDiffusedAttention, self).__init__()

        self.head_count = head_count
        self.model_dim = model_dim
        self.d_k = d_k
        self.d_v = d_v

        self.key = nn.Linear(model_dim, head_count * self.d_k)
        self.query = nn.Linear(model_dim, head_count * self.d_k)
        self.value = nn.Linear(model_dim, head_count * self.d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.head_count * d_v, model_dim)
        self.diffuse = diffuse
        self.alpha = alpha

    def forward(self, key, value, query, mask=None, layer_cache=None,
                attn_type=None, step=None):

        batch_size = key.size(0)
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        use_gpu = key.is_cuda

        def shape(x, dim):
            """  projection """
            return x.view(batch_size, -1, head_count, dim).transpose(1, 2)

        def unshape(x, dim):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim)

        # 1) Project key, value, and query.
        key = shape(self.key(key), self.d_k)
        value = shape(self.value(value), self.d_v)
        query = shape(self.query(query), self.d_k)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(self.d_k)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        scores = query_key.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)

        drop_attn = self.dropout(attn)
        context_original = torch.matmul(drop_attn, value)

        # --------- Diffuse attn approx, eq (5) ---------
        if self.diffuse!=0:
            z0=value
            context_diffused=context_original
            for i in range(self.diffuse):
                context_original=(1-self.alpha[0])*context_diffused+self.alpha[0]*z0
                context_diffused = torch.matmul(drop_attn, context_original)
        #

        context = unshape(context_original, self.d_v)
        final_output = self.output(context)

        # a list of size num_heads containing tensors
        # of shape `batch x query_len x key_len`
        attn_per_head = [attn.squeeze(1)
                         for attn in attn.chunk(head_count, dim=1)]
        return final_output, attn_per_head

    def update_dropout(self, dropout):
        self.dropout.p = dropout


""" Encoder """

class EncoderBase(nn.Module):
    def _check_args(self, src, mask=None, hidden=None):
        n_batch, _, _ = src.size()
        if mask is not None:
            n_batch_, _, _ = mask.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, mask=None):
        raise NotImplementedError

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, d_k, d_v, dropout, diffuse, alpha):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadedDiffusedAttention(heads, d_model, d_k, d_v,
                            dropout=dropout, diffuse=diffuse, alpha=alpha)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask):
        context, attn_per_head = self.attention(inputs, inputs, inputs,
                                    mask=mask, attn_type="self")
        out = self.layer_norm(self.dropout(context) + inputs)
        return self.feed_forward(out), attn_per_head

class TransformerEncoder(EncoderBase):
    def __init__(self, num_layers, d_model=512, heads=8, d_k=64, d_v=64,
                 d_ff=2048, dropout=0.2, diffuse=0, alpha=0.15):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, d_k, d_v,
                dropout, diffuse, alpha) for i in range(num_layers)])

    def count_parameters(self):
        params = list(self.layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, src, mask=None):
        self._check_args(src, mask)
        out = src
        # Run the forward pass of every layer of the tranformer.
        representations = []
        attention_scores = []
        for i in range(self.num_layers):
            out, attn_per_head = self.layer[i](out, mask)
            representations.append(out)
            attention_scores.append(attn_per_head)
        return representations, attention_scores

