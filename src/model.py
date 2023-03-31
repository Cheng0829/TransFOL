
import enum
from typing import Callable, List, Union
import numpy as np
import torch, time
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from math import sin, cos
from typing import Optional
from graph_util import Graph, GraphEmbed, GraphWithAnswer, BatchMatGraph

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
 
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim),requires_grad=True).cuda()
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
        Args: 
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        """
        GCN1:
        input_feature: feat[0]: [7,1024]
        self.weight: [1024,1710]
        support: [7,1710]
        adjacency: [1710,1710]
        output: [7,1710]

        GCN2:
        input_feature: [7,1710]
        self.weight: [1710,1710]
        support: [7,1710]
        adjacency: [1710,1710]
        output: [7,1710]
        """
        
        # GCN1: support: [adjacency_size, hidden_size] * [hidden_size, adjacency_size], 
        # GCN2: support: [adjacency_size, adjacency_size] * [adjacency_size, hidden_size]
        support = torch.mm(input_feature, self.weight) 
        output = torch.sparse.mm(adjacency, support)
        output = output + self.bias
        return output # [7,1710]

class GcnNet(nn.Module):
    def __init__(self, input_dim, adjacency_size, out_dim):
        super().__init__()
        """3层GCN"""
        self.gcn1 = GraphConvolution(input_dim, adjacency_size)
        self.gcn2 = GraphConvolution(adjacency_size, adjacency_size)
        self.gcn3 = GraphConvolution(adjacency_size, out_dim)
        self.relu = nn.ReLU()

    def forward(self, adjacency, feat, data):
        adjacency_matrix = torch.zeros(len(adjacency),feat.size(2), requires_grad=True).cuda()
        feature = torch.zeros(feat.size(0),feat.size(1)//7*2,feat.size(2), requires_grad=True).cuda()
        for i in range(len(feat)):
            feature[i][0] = feat[i][0]
            feature[i][1] = feat[i][1] # feature: [batch_Size, 2, 1024]
        feature = feature.view(-1, feat.size(2)) # [batch_Size*2, hidden_size] [4096, 1024]
        for i in range(len(data.x)//7):
            e1 = data.x[7*i]
            adjacency_matrix[e1] = adjacency_matrix[e1] + feature[2*i]
            e2 = data.x[7*i+1]
            adjacency_matrix[e2] = adjacency_matrix[e2] + feature[2*i+1] # adjacency_matrix: [num_nodes, hidden_size]

        logits = self.relu(self.gcn1(adjacency, adjacency_matrix)) # [num_nodes, num_nodes]
        logits = self.gcn2(adjacency, logits) # [num_nodes, hidden_size]
        logits = self.gcn3(adjacency, logits) # [num_nodes, hidden_size]
        
        for i in range(len(data.x)//7):
            e1 = data.x[7*i]
            feature[2*i] = logits[e1]
            e2 = data.x[7*i+1]
            feature[2*i+1] = logits[e2] # adjacency_matrix: [num_nodes, hidden_size]
        feature = feature.view(-1, 2, feature.size(1))
        feat[::, 0, ::] = feature[::, 1, ::]
        feat[::, 1, ::] = feature[::, 0, ::]
        return feat

class TokenEmbedding(Module):
    r"""
    Hold separate embeddings for different types of tokens and adds type embeddings to the tokens.
    为不同类型的token保留不同的嵌入并把类型嵌入加入标记。
    """

    def __init__(self, embed_dim, embed_value: List[Union[int, torch.nn.Embedding, Module]]):
        r"""
        :param embed_dim: The number of features for each node
        :param embed_value: A list containing num_nodes, the embedding dict, or mixed, for each type.
        :param embed_dim:每个节点的特征数量
        :param embed_value:包含每种类型的节点数量和嵌入字典的混合列表。
        """
        super().__init__()
        self.embed_token = []
        """D_KGTransformer
        self.token_embed = TokenEmbedding(self.d_model, embed_value=[self.num_nodes(1710), 1, relation_cnt(43) * 2])
        """
        for i, item in enumerate(embed_value): 
            """
            i item:  0, 1710    1, 1    2, 86
            """
            if isinstance(item, int):
                '''item = [1710,512], [1,512], [86,512]'''
                item = torch.nn.Embedding(item, embed_dim) 
                """**********************************************************************************************"""
                # item = torch.nn.Embedding(14429, embed_dim)
                """**********************************************************************************************"""
            elif isinstance(item, torch.nn.Embedding):
                assert item.embedding_dim == embed_dim
            self.add_module(f'embed_token_{i}', item)
            self.embed_token.append(item)
        self.embed_type = torch.nn.Embedding(len(embed_value), embed_dim) # [3,512]


    def forward(self, node_type, node_id) -> torch.FloatTensor:
        """D_KGTransformer
        feat = self.token_embed(data.embed_type, data.x) # feat = self.token_embed([3584],[3584])
        """
        # Node type embedding as a base
        # 节点类型嵌入作为基础
        '''
        self.embed_type: [3,512]
        node_type: [3584]    [0,0,1,2,2,2,2, ......]
        node_id: [3584]    即data.x [10,1,0,0,0,43,43, ......]
        '''
        feat = self.embed_type(node_type) # [3,512]([3584]) -> [3584,512]
        for i, embed in enumerate(self.embed_token):

            """把类型信息加入token的嵌入"""
            mask = node_type == i
            # Add token embedding 添加令牌嵌入
            # TODO: check whether the in-place operation works with gradients 检查就地操作是否适用于梯度
            
            # 报错原因:实体有14429个, 最大的编号为14428, 但是数据集中并没有14429个实体,导致嵌入矩阵维度不够
            """
            i, embed:      0,[1710,512]      1,[1,512]      2,[86,512]
            """
            """三元组变换后把关系也当作实体表示, 所以和首尾实体共用一个嵌入矩阵"""
            feat[mask] += embed(node_id[mask]) # [3584,512]
            """"""
        
        j = 0
        while j < len(node_type)/7:
            # break
            """聚集关系数据,因此不能用在DDI等单关系实验上"""
            e_1 = node_id[7*j]
            e_2 = node_id[7*j+1]
            r_1 = node_id[7*j+3]
            r_2 = node_id[7*j+4]
            r_3 = node_id[7*j+5]
            r_4 = node_id[7*j+6]
            if max(e_1,e_2,r_1,r_2,r_3,r_4) < len(feat)//7:

                """新MSTE Sin"""
                # tmp_e_1 = torch.Tensor.detach(torch.sin(0.5*(feat[r_1]+feat[r_3]) * feat[e_2])) * feat[e_1]
                # tmp_e_2 = torch.Tensor.detach(torch.sin(0.5*(feat[r_4]+feat[r_2]) * feat[e_1])) * feat[e_2]
                # tmp_r_1 = torch.Tensor.detach(torch.sin(feat[e_1] * feat[r_3])) * feat[r_1]
                # tmp_r_2 = torch.Tensor.detach(torch.sin(feat[e_2] * feat[r_4])) * feat[r_2]
                # tmp_r_3 = torch.Tensor.detach(torch.sin(feat[e_1] * feat[r_1])) * feat[r_3]
                # tmp_r_4 = torch.Tensor.detach(torch.sin(feat[e_2] * feat[r_2])) * feat[r_4]
                # feat[e_1] = tmp_e_1
                # feat[e_2] = tmp_e_2
                # feat[r_1] = tmp_r_1
                # feat[r_2] = tmp_r_2
                # feat[r_3] = tmp_r_3
                # feat[r_4] = tmp_r_4
                
                """12.TEMP"""
                # Ge1_r1 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[e_1], feat[r_1])).cuda()
                # Ge1_e2 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[e_1], feat[e_2])).cuda()
                # Ge2_r2 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[e_2], feat[r_2])).cuda()
                # Ge2_e1 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[e_2], feat[e_1])).cuda()

                # Gr1_e1 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_1], feat[e_1])).cuda()
                # Gr1_r2 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_1], feat[r_2])).cuda()
                # Gr2_e2 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_2], feat[e_2])).cuda()
                # Gr2_r1 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_2], feat[r_1])).cuda()

                # feat[e_1] = torch.Tensor.detach(HighwayNetwork(hidden_size=feat.size(-1))(Ge1_r1, Ge1_e2)).cuda()
                # feat[e_2] = torch.Tensor.detach(HighwayNetwork(hidden_size=feat.size(-1))(Ge2_r2, Ge2_e1)).cuda()
                # feat[r_1] = torch.Tensor.detach(HighwayNetwork(hidden_size=feat.size(-1))(Gr1_e1, Gr1_r2)).cuda()
                # feat[r_2] = torch.Tensor.detach(HighwayNetwork(hidden_size=feat.size(-1))(Gr2_e2, Gr2_r1)).cuda()

                """新12.TEMP"""
                # Ge1_r1 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[e_1], feat[r_1])).cuda()
                # Ge1_r3 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[e_1], feat[r_3])).cuda()
                # Ge2_r2 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[e_2], feat[r_2])).cuda()
                # Ge2_r4 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[e_2], feat[r_4])).cuda()

                # Gr1_e1 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_1], feat[e_1])).cuda()
                # Gr1_r2 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_1], feat[r_2])).cuda()
                # Gr2_e2 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_2], feat[e_2])).cuda()
                # Gr2_r1 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_2], feat[r_1])).cuda()

                # """接下来应该尝试r3,r4"""
                # Gr3_e1 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_3], feat[e_1])).cuda()
                # Gr3_r4 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_3], feat[r_4])).cuda()
                # Gr4_e2 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_4], feat[e_2])).cuda()
                # Gr4_r3 = torch.Tensor.detach(Match(hidden_size=feat.size(-1))(feat[r_4], feat[r_3])).cuda()

                # feat[e_1] = torch.Tensor.detach(HighwayNetwork(hidden_size=feat.size(-1))(Ge1_r1, Ge1_r3)).cuda()
                # feat[e_2] = torch.Tensor.detach(HighwayNetwork(hidden_size=feat.size(-1))(Ge2_r2, Ge2_r4)).cuda()
                # feat[r_1] = torch.Tensor.detach(HighwayNetwork(hidden_size=feat.size(-1))(Gr1_e1, Gr1_r2)).cuda()
                # feat[r_2] = torch.Tensor.detach(HighwayNetwork(hidden_size=feat.size(-1))(Gr2_e2, Gr2_r1)).cuda()

                # """***"""
                # feat[r_3] = torch.Tensor.detach(HighwayNetwork(hidden_size=feat.size(-1))(Gr3_e1, Gr3_r4)).cuda()
                # feat[r_4] = torch.Tensor.detach(HighwayNetwork(hidden_size=feat.size(-1))(Gr4_e2, Gr4_r3)).cuda()

            j = j + 1

        return feat

class Match(nn.Module):
    def __init__(self, hidden_size): # hidden_size=400
        super().__init__()
        self.trans_linear = nn.Linear(hidden_size, hidden_size).cuda()
        self.map_linear = nn.Linear(2*hidden_size, 2*hidden_size).cuda()
        self.relu = nn.ReLU().cuda()
    def forward(self, proj_p, proj_q):
        # proj_p, proj_q: [hidden_size,]
        # 一个FC层
        # '''论文中的双向注意力机制'''
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p * trans_q# proj_p.bmm(torch.transpose(trans_q, 1, 2))
        att_norm = torch.nn.functional.softmax(att_weights, dim=-1)
        att_vec = att_norm * proj_q # .bmm(proj_q)

        # trans_q = self.trans_linear(proj_q)
        # att_weights = proj_p.bmm(torch.transpose(trans_q, 0, 1))
        # att_norm = torch.nn.functional.softmax(att_weights, dim=-1)
        # att_vec = att_norm.bmm(proj_q)

        '''公式7/8'''
        elem_min = att_vec - proj_p
        elem_mul = att_vec * proj_p
        all_con = torch.cat([elem_min, elem_mul])
        output = self.relu(self.map_linear(all_con))
        return output # Ger 2*hidden_size

class HighwayNetwork(nn.Module):
    def __init__(self, hidden_size, with_sigmoid=False):
        super().__init__()
        self.linear = nn.Linear(2*hidden_size, hidden_size).cuda()
        self.linear2 = nn.Linear(2*hidden_size, 2*hidden_size).cuda()
        self.linear3 = nn.Linear(2*hidden_size, 2*hidden_size).cuda()
        self.with_sigmoid =  with_sigmoid
        self.sigmoid = nn.Sigmoid().cuda()
    def forward(self, p, q):
        # inputs: Ger Ges -> embedding
        lp = self.linear2(p) # Ger
        lq = self.linear3(q) # Ges
        '''公式9'''
        sigma = self.sigmoid(lp+lq) # 门控单元
        '''公式10'''
        output = sigma*p + (1-sigma)*q # Ge
        '''公式11'''
        output = self.linear(output) # He
        if self.with_sigmoid:
            output = torch.sigmoid(output).cuda()
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_num):
        super().__init__()

        self.head_num = head_num

        self.att_size = att_size = hidden_size // head_num
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_num * att_size)
        self.linear_k = nn.Linear(hidden_size, head_num * att_size)
        self.linear_v = nn.Linear(hidden_size, head_num * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_num * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):

        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        # print('多头注意力batch_size:',q.size(0)) # 2
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_num, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_num, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_num, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_num * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x

class entity_EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_num):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_num)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.attn_bias = nn.Parameter(torch.Tensor(2,2))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.zeros_(self.attn_bias)

    def forward(self, x, attn_bias=None):
        # x, y: [1024, 7, 1024] [batch_size, 7, hidden_size]
        y = self.self_attention_norm(x)
        # y = self.self_attention(y, y, y, attn_bias)
        y[::, 0:2, ::] = self.self_attention(y[::, 0:2, ::], y[::, 0:2, ::], y[::, 0:2, ::], attn_bias=self.attn_bias)
        # y[::, 3:, ::] = self.self_attention(y[::, 3:, ::], y[::, 3:, ::], y[::, 3:, ::], attn_bias=None)
        y = self.self_attention_dropout(y)
        x = x + y # 残差
        y = self.ffn_norm(x) # 正则化
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class relation_EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_num):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_num)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.attn_bias = nn.Parameter(torch.Tensor(4,4))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.zeros_(self.attn_bias)
    def forward(self, x, attn_bias=None):
        # x, y: [1024, 7, 1024] [batch_size, 7, hidden_size]
        y = self.self_attention_norm(x)
        # y = self.self_attention(y, y, y, attn_bias)
        y[::, 3:, ::] = self.self_attention(y[::, 3:, ::], y[::, 3:, ::], y[::, 3:, ::], attn_bias=self.attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y # 残差
        y = self.ffn_norm(x) # 正则化
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class cross_entity_EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_num):
        super().__init__()
        self.cross_self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_num)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_num)
        self.attention_dropout_first = nn.Dropout(dropout_rate)
        self.attention_dropout_second = nn.Dropout(dropout_rate)
        self.attention_norm_first = nn.LayerNorm(hidden_size)
        self.attention_norm_second = nn.LayerNorm(hidden_size)
        self.ffn_attention_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.attn_bias = nn.Parameter(torch.Tensor(2,2))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.zeros_(self.attn_bias)

    def forward(self, x, attn_bias=None):
        '''多头注意力1'''
        y = self.attention_norm_first(x)
        y[::, 0:2, ::] = self.cross_self_attention(y[::, 0:2, ::], y[::, 3:5, ::], y[::, 5:, ::], attn_bias=self.attn_bias)
        y = self.attention_dropout_first(y)
        x = x + y # 残差
        '''多头注意力2''' 
        y = self.attention_norm_second(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.attention_dropout_second(y)
        x = x + y
        '''FFN'''
        x = self.ffn_attention_norm(x) # pre LayerNorm
        y = self.ffn(x)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class cross_relation_EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_num):
        super().__init__()
        self.cross_self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_num)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_num)
        self.attention_dropout_first = nn.Dropout(dropout_rate)
        self.attention_dropout_second = nn.Dropout(dropout_rate)
        self.attention_norm_first = nn.LayerNorm(hidden_size)
        self.attention_norm_second = nn.LayerNorm(hidden_size)
        self.ffn_attention_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.attn_bias = nn.Parameter(torch.Tensor(2,2))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.zeros_(self.attn_bias)

    def forward(self, x, attn_bias=None):
        '''多头注意力1'''
        y = self.attention_norm_first(x)
        y[::, 3:5, ::] = self.self_attention(y[::, 3:5, ::], y[::, 5:, ::], y[::, 0:2, ::], attn_bias=self.attn_bias)
        y = self.attention_dropout_first(y)
        x = x + y # 残差
        '''多头注意力2''' 
        y = self.attention_norm_second(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.attention_dropout_second(y)
        x = x + y
        '''FFN'''
        x = self.ffn_attention_norm(x) # pre LayerNorm
        y = self.ffn(x)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class D_KGTransformer(Module):
    class TokenType(enum.IntEnum):
        """ node_type: [0,0,1,2,2,2,2] -> node_id/data.x: [10,1,0,0,0,43,43, ......]"""
        """ node_id/data.x的前0和1列是2i查询实体, 第3和4列是2i查询关系, 最后两列是relation_cnt+2i查询关系 """
        Ent = 0
        MaskEnt = 1
        Rel = 2
        MaskRel = 1

    def __init__(self, num_nodes: int, relation_cnt: int, config):
        super().__init__()
        self.d_model = config['hidden_size']
        self.num_nodes = num_nodes
        self.num_heads = config['num_heads']
        self.relation_cnt = relation_cnt
        # Check all appearances of token_embed before changing the scheme!
        # self.num_nodes: 1710(实体总数), relation_cnt: 43(使用时*2=86为答案类别)
        self.token_embed = TokenEmbedding(self.d_model, embed_value=[
            self.num_nodes,
            1,
            relation_cnt * 2,
        ])
        self.attn_bias_embed = nn.Embedding(40, self.num_heads, padding_idx=1)
        with torch.no_grad():
            self.attn_bias_embed.weight[1] = torch.full((self.num_heads,), float('-inf'))


        self.entity_encode_layers = nn.ModuleList([
            entity_EncoderLayer(
                hidden_size=config['hidden_size'],
                ffn_size=config['dim_feedforward'],
                dropout_rate=config['dropout'],
                attention_dropout_rate=config['attention_dropout'],
                head_num=config['num_heads']
            )
            for _ in range(2)
        ])
        self.relation_encode_layers = nn.ModuleList([
            relation_EncoderLayer(
                hidden_size=config['hidden_size'],
                ffn_size=config['dim_feedforward'],
                dropout_rate=config['dropout'],
                attention_dropout_rate=config['attention_dropout'],
                head_num=config['num_heads']
            )
            for _ in range(1)
        ])

        self.cross_entity_encode_layers = nn.ModuleList([
            cross_entity_EncoderLayer(
                hidden_size=config['hidden_size'],
                ffn_size=config['dim_feedforward'],
                dropout_rate=config['dropout'],
                attention_dropout_rate=config['attention_dropout'],
                head_num=config['num_heads']
            )
            for _ in range(8)
        ])

        self.cross_relation_encode_layers = nn.ModuleList([
            cross_relation_EncoderLayer(
                hidden_size=config['hidden_size'],
                ffn_size=config['dim_feedforward'],
                dropout_rate=config['dropout'],
                attention_dropout_rate=config['attention_dropout'],
                head_num=config['num_heads']
            )
            for _ in range(0)
        ])

        self.cat_linear = torch.nn.Linear(config['hidden_size'], config['hidden_size'])

        """GCN"""
        self.adjacency = torch.zeros(num_nodes, num_nodes).cuda()
        self.gcnnet = GcnNet(input_dim=config['hidden_size'], adjacency_size=num_nodes, out_dim=config['hidden_size']) # 

        self.final_ln = nn.LayerNorm(self.d_model)
        self.pred_ent_proj = torch.nn.Linear(self.d_model, self.num_nodes)
        self.loss_type = config['loss']
        self.smoothing = config['smoothing']        

    def forward(self, data: BatchMatGraph):
        # self.attn_bias_embed: [40,8]    data.attn_bias_type: [1024,7,7]
        rel_pos_bias = self.attn_bias_embed(data.attn_bias_type) # rel_pos_bias: [1024,8,7,7]
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        rel_pos_bias = rel_pos_bias.permute(0, 3, 1, 2)
        attn_bias = rel_pos_bias

        feat = self.token_embed(data.embed_type, data.x) # ([3584],[3584]) -> [3584,512]
        
        feat = feat.view(data.num_graphs, data.num_nodes_per_graph, self.d_model) # [512,7,512] [batch_size, 7, hidden_size]

        """e12 Encoder"""
        feat_e = feat
        for layer in self.entity_encode_layers:
            feat_e = layer(feat_e)

        """r1234 Encoder"""
        feat_r = feat
        for layer in self.relation_encode_layers:
            feat_r = layer(feat_r)

        """cat"""
        feat = torch.cat([feat_e[::,0:3,::], feat_r[::,3:,::]], dim=1)
        feat_e = feat
        feat_r = feat

        """cross Encoder + Encoder"""
        for layer in self.cross_entity_encode_layers:
            feat_e = layer(feat_e, attn_bias)
        for layer in self.cross_relation_encode_layers:
            feat_r = layer(feat_r, attn_bias)

        """cat"""
        feat = self.cat_linear(torch.cat([feat_e[::,0:3,::], feat_r[::,3:,::]], dim=1))

        '''*****************************************************************************************************************'''

        """GCN"""
        for i in range(0, len(data.x), 7):
            row = data.x[i]
            column = data.x[i+1]
            self.adjacency[row][column] = 1
        feat_gcn = self.gcnnet(self.adjacency, feat, data)
        feat = feat_gcn
        
        '''*****************************************************************************************************************'''
        
        feat = self.final_ln(feat)
        feat = feat.view(-1, self.d_model)
        return feat

    def answer_queries(self, data: BatchMatGraph):
        r"""
        :param data: BatchMatGraph
        :return:
        """
        e_pred = None
        o_pred = None
        feat = self(data)
        device = data.x.device
        relabel_arr = torch.empty(data.x.shape, dtype=torch.long, device=device)
        # Currently supports query type 0 (entities) only
        mask = data.pred_type == 0
        mask_cnt = torch.count_nonzero(mask).item()

        # relabel all the nodes
        relabel_arr[mask] = torch.arange(mask_cnt, device=device)

        if min(data.joint_nodes.shape) != 0:
            sfm = torch.nn.Softmax(dim=1)
            q_mask = mask[data.x_query]
            jq_mask = mask[data.joint_nodes]
            uq_mask = mask[data.union_query]
            p_mask = mask[data.pos_x]

            x_pred = self.pred_ent_proj(feat[mask])
            x_pred = x_pred.double()
            jq = data.joint_nodes[jq_mask]
            uq = data.union_query[uq_mask]
            assert sum(jq) == sum(data.joint_nodes)
            assert sum(uq) == sum(data.union_query)

            relabeled_jq_even = relabel_arr[jq[::2]]
            relabeled_jq_odd = relabel_arr[jq[1::2]]
            relabeled_uq = relabel_arr[uq]

            x_pred[relabeled_jq_even] = sfm(x_pred[relabeled_jq_even])
            x_pred[relabeled_jq_odd] = sfm(x_pred[relabeled_jq_odd])
            q_score = None
            if data.x_ans is not None:
                # q_score = torch.max(x_pred[relabeled_jq_even, data.x_ans], x_pred[relabeled_jq_odd, data.x_ans])
                e_score = x_pred[relabeled_jq_even, data.x_ans]
                o_score = x_pred[relabeled_jq_odd, data.x_ans]

            # Mask out all positive answers (including the predicted one)
            x_pred[relabel_arr[data.pos_x[p_mask]], data.pos_ans[p_mask]] = float('-inf')
            if data.x_ans is not None:
                x_pred[relabeled_jq_even, data.x_ans] = e_score
                x_pred[relabeled_jq_odd, data.x_ans] = o_score

            # Using rank as score
            eind = torch.argsort(x_pred[relabeled_jq_even], dim=1)
            fi = torch.arange(x_pred[relabeled_jq_even].shape[1], dtype=x_pred.dtype, device=x_pred.device).repeat(
                x_pred[relabeled_jq_even].shape[0], 1)
            x_pred[relabeled_jq_even] = torch.scatter(x_pred[relabeled_jq_even], 1, eind, fi)

            oind = torch.argsort(x_pred[relabeled_jq_odd], dim=1)
            fi2 = torch.arange(x_pred[relabeled_jq_odd].shape[1], dtype=x_pred.dtype, device=x_pred.device).repeat(
                x_pred[relabeled_jq_odd].shape[0], 1)
            x_pred[relabeled_jq_odd] = torch.scatter(x_pred[relabeled_jq_odd], 1, oind, fi2)

            q_pred = torch.max(x_pred[relabeled_jq_even], x_pred[relabeled_jq_odd])
            e_pred = x_pred[relabeled_jq_even]
            o_pred = x_pred[relabeled_jq_odd]
        else:
            # q_mask and p_mask: queries on entities (should all be True)
            q_mask = mask[data.x_query]
            p_mask = mask[data.pos_x]

            # predict for all the nodes
            x_pred = self.pred_ent_proj(feat[mask])

            # relabel the query
            relabeled_query = relabel_arr[data.x_query[q_mask]]

            # If we are training, we have to make sure that answers are not masked
            q_score = None
            if data.x_ans is not None:
                q_score = x_pred[relabeled_query, data.x_ans[q_mask]]

            # Mask out all positive answers (including the predicted one)
            x_pred[relabel_arr[data.pos_x[p_mask]], data.pos_ans[p_mask]] = float('-inf')
            q_pred = x_pred[relabeled_query]

            # Add back those to be predicted so that we know the scores of the x_ans
            if q_score is not None:
                q_pred[torch.arange(q_mask.shape[0], device=device), data.x_ans[q_mask]] = q_score
        return q_pred, None, None 

    def forward_loss(self, data: BatchMatGraph):
        feat = self(data)
        device = data.x.device
        relabel_arr = torch.empty(data.x.shape, dtype=torch.long, device=device)
        # Currently supports query type 0 (entities) only
        mask = data.pred_type == 0
        mask_cnt = torch.count_nonzero(mask).item()

        # relable all the nodes
        relabel_arr[mask] = torch.arange(mask_cnt, device=device)

        from metric import loss_cross_entropy_multi_ans, loss_label_smoothing_multi_ans
        q_mask = mask[data.x_query]
        p_mask = mask[data.pos_x]
        if self.loss_type == "CE":
            f = feat[mask]
            l, w = loss_cross_entropy_multi_ans(
                self.pred_ent_proj(f).double(),
                relabel_arr[data.x_query[q_mask]], data.x_ans[q_mask],
                relabel_arr[data.pos_x[p_mask]], data.pos_ans[p_mask],
                query_w=data.x_pred_weight[q_mask],
            )
            
        elif self.loss_type == 'LS':
            f = feat[mask]
            l, w = loss_label_smoothing_multi_ans(
                self.pred_ent_proj(f).double(),
                relabel_arr[data.x_query[q_mask]], data.x_ans[q_mask],
                relabel_arr[data.pos_x[p_mask]], data.pos_ans[p_mask],
                self.smoothing,
                query_w=data.x_pred_weight[q_mask]
            )
        import math
        # assert not math.isnan(l.item())
        return l, w # 即loss, weight_sum

class D_KGTransformerLoss(Module):
    def __init__(self, model: Module):
        super(D_KGTransformerLoss, self).__init__()
        self.model = model

    def forward(self, data):
        return self.model.forward_loss(data)
