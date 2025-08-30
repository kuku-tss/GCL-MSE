import dgl
import math
import torch as th
import torch.nn as nn
from torch.nn import init
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F

from utils import get_activation, to_etype_name
from torch.nn.parameter import Parameter

th.set_printoptions(profile="full")

class GraphConvolution(nn.Module):
    """简单GCN层"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = th.mm(input, self.weight)
        output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, features, nhid, nhid2, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(features, nhid)
        self.gc2 = GraphConvolution(nhid, nhid2)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class FGCN(nn.Module):
    def __init__(self, fdim_drug, fdim_disease, nhid1, nhid2, dropout):
        super(FGCN, self).__init__()
        self.FGCN1 = GCN(fdim_drug, nhid1, nhid2, dropout)
        self.FGCN2 = GCN(fdim_disease, nhid1, nhid2, dropout)
        self.dropout = dropout

    def forward(self, drug_graph, drug_sim_feat, dis_graph, disease_sim_feat):
        emb1 = self.FGCN1(drug_sim_feat, drug_graph)
        emb2 = self.FGCN2(disease_sim_feat, dis_graph)
        return emb1, emb2


class GCMCGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None, Two_Stage=False):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat  # unpack
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise dgl.DGLError('External weight is provided while at the same time the'
                                       ' module has defined its own weight parameter. Please'
                                       ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)

            feat = feat * self.dropout(cj)
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = rst * ci

        return rst


class GCMCLayer(nn.Module):
    def __init__(self, rating_vals,
                 user_in_units,
                 movie_in_units,
                 msg_units,
                 out_units,
                 dropout_rate=0.0,
                 agg='stack',
                 agg_act=None,
                 share_user_item_param=False,
                 basis_units=4, device=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_user_item_param = share_user_item_param
        self.ufc = nn.Linear(msg_units, out_units)
        self.user_in_units = user_in_units
        self.msg_units = msg_units
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
        if agg == 'stack':
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)

        msg_units = msg_units // 3
        self.msg_units = msg_units
        self.dropout = nn.Dropout(dropout_rate)
        
        # 初始化一个标志，用于跟踪是否使用权重
        self.has_weights = False
        
        self.basis_units = basis_units
        self.att = nn.Parameter(th.randn(len(self.rating_vals), basis_units))
        self.basis = nn.Parameter(th.randn(basis_units, user_in_units, msg_units))
        
        subConv = {}
        for i, rating in enumerate(rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            if share_user_item_param and user_in_units == movie_in_units:
                self.has_weights = True
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=False,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(user_in_units,
                                                    msg_units,
                                                    weight=False,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
            else:
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=True,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(movie_in_units,
                                                    msg_units,
                                                    weight=True,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        """Put parameters into device except W_r."""
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_user_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, drug_feat=None, dis_feat=None, Two_Stage=False):
        in_feats = {'drug': drug_feat, 'disease': dis_feat}
        mod_args = {}
        
        # 使用批处理张量乘法优化
        self.W = th.einsum('rb,bio->rio', self.att, self.basis)
        
        # 为每个评级类型准备参数
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            
            weight = self.W[i, :, :] if self.has_weights else None
            
            mod_args[rating] = (weight, Two_Stage)
            mod_args[rev_rating] = (weight, Two_Stage)

        # 应用异构图卷积
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        drug_feat = out_feats['drug']
        dis_feat = out_feats['disease']

        if in_feats['disease'].shape == dis_feat.shape:
            ufeat = dis_feat.view(dis_feat.shape[0], -1)
            ifeat = drug_feat.view(drug_feat.shape[0], -1)

        # 应用激活函数
        if self.agg_act is not None:
            drug_feat = self.agg_act(drug_feat)
            dis_feat = self.agg_act(dis_feat)
        
        # 应用dropout
        drug_feat = self.dropout(drug_feat)
        dis_feat = self.dropout(dis_feat)
        
        # 应用线性变换
        drug_feat = self.ifc(drug_feat)
        dis_feat = self.ufc(dis_feat)

        return drug_feat, dis_feat



# 简单MLP解码器
class SimpleMLPDecoder(nn.Module):
    def __init__(self, in_units, hidden_units=128, dropout_rate=0.2):
        super(SimpleMLPDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_units * 4, hidden_units),
            nn.LayerNorm(hidden_units),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.LayerNorm(hidden_units // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units // 2, 1)
        )
    
    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat
            
            graph.apply_edges(udf_u_mul_e)
            edge_feat = graph.edata['m']  # [num_edges, drug_dim + dis_dim]
            
            # 直接通过MLP处理边特征
            pred = self.decoder(edge_feat)
            
        return pred
    

    

    
class ImprovedMLPDecoder(nn.Module):
    def __init__(self, in_units, hidden_units=128, dropout_rate=0.2):
        super(ImprovedMLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_units * 2, hidden_units)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units // 2)
        self.norm2 = nn.LayerNorm(hidden_units // 2)
        self.fc3 = nn.Linear(hidden_units // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        # 残差连接投影层
        self.shortcut = nn.Linear(in_units * 2, hidden_units // 2)
        
    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat
            
            # 用 enhanced_edge_features 计算边特征
            graph.apply_edges(udf_u_mul_e)
            x = graph.edata['m']
            
            h = self.fc1(x)
            h = self.norm1(h)
            h = F.gelu(h)
            h = self.dropout(h)
            
            identity = self.shortcut(x)
            h = self.fc2(h)
            h = self.norm2(h)
            h = F.gelu(h + identity)
            h = self.dropout(h)
            
            out = self.fc3(h)
            
        return out
    
class LightweightMLPDecoder(nn.Module):
    def __init__(self, in_units, hidden_units=64, dropout_rate=0.3):
        super(LightweightMLPDecoder, self).__init__()
        # 简化为两层，减少参数量
        self.fc1 = nn.Linear(in_units * 2, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)
        
        # 增强正则化
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(hidden_units)
        
        # 权重初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat
            
            # 保持原有的边特征计算
            graph.apply_edges(udf_u_mul_e)
            x = graph.edata['m']
            
            # 简化的前向传播
            h = self.fc1(x)
            h = self.bn(h)
            h = F.relu(h)  # 使用更稳定的ReLU
            h = self.dropout(h)
            
            out = self.fc2(h)
            
        return out



class MLPDecoder(nn.Module):
    def __init__(self,
                 in_units,
                 dropout_rate=0.2):
        super(MLPDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

        self.lin1 = nn.Linear(4 * in_units, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat
            graph.apply_edges(udf_u_mul_e)
            out = graph.edata['m']

            out = F.relu(self.lin1(out))
            out = self.dropout(out)

            out = F.relu(self.lin2(out))
            out = self.dropout(out)

            # out = self.sigmoid(self.lin3(out))
            out = self.lin3(out)
        return out

# 主网络
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.layers = args.layers
        self._act = get_activation(args.model_activation)
        
        # GCN层
        self.TGCN = nn.ModuleList()
        self.TGCN.append(GCMCLayer(args.rating_vals,
                                   args.src_in_units,
                                   args.dst_in_units,
                                   args.gcn_agg_units,
                                   args.gcn_out_units,
                                   args.dropout,
                                   args.gcn_agg_accum,
                                   agg_act=self._act,
                                   share_user_item_param=args.share_param,
                                   device=args.device))
        
        # 模型参数
        self.gcn_agg_accum = args.gcn_agg_accum
        self.rating_vals = args.rating_vals
        self.device = args.device
        self.gcn_agg_units = args.gcn_agg_units
        self.src_in_units = args.src_in_units
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(args.gcn_out_units) for _ in range(args.layers)
        ])
        
        # 更多GCN层
        for i in range(1, args.layers):
            if args.gcn_agg_accum == 'stack':
                gcn_out_units = args.gcn_out_units * len(args.rating_vals)
            else:
                gcn_out_units = args.gcn_out_units
                
            self.TGCN.append(GCMCLayer(args.rating_vals,
                                       args.gcn_out_units,
                                       args.gcn_out_units,
                                       gcn_out_units,
                                       args.gcn_out_units,
                                       args.dropout,
                                       args.gcn_agg_accum,
                                       agg_act=self._act,
                                       share_user_item_param=args.share_param,
                                       device=args.device))

        # 使用FGCN
        self.FGCN = FGCN(
            args.fdim_drug,
            args.fdim_disease,
            args.nhid1,
            args.nhid2,
            args.dropout
        )

        
        # 使用简单MLP解码器
        self.decoder = LightweightMLPDecoder(
            in_units=args.gcn_out_units, 
            # hidden_units=128, 
            # dropout_rate=args.dropout
        )

    def forward(self, enc_graph, dec_graph,
                drug_graph, drug_sim_feat, drug_feat,
                dis_graph, disease_sim_feat, dis_feat,
                Two_Stage=False):
        
        # 拓扑卷积操作
        drug_out, dis_out = None, None
        for i in range(0, self.layers):
            drug_o, dis_o = self.TGCN[i](enc_graph, drug_feat, dis_feat, Two_Stage)
            
            # 应用残差连接和层归一化
            if i > 0:
                drug_o = self.layer_norms[i](drug_o + drug_feat)
                dis_o = self.layer_norms[i](dis_o + dis_feat)
            
            # 累积特征
            if i == 0:
                drug_out = drug_o
                dis_out = dis_o
            else:
                drug_out += drug_o
                dis_out += dis_o
            
            # 更新特征
            drug_feat = drug_o
            dis_feat = dis_o

        # 特征卷积操作
        drug_sim_out, dis_sim_out = self.FGCN(
            drug_graph, drug_sim_feat, dis_graph, disease_sim_feat
        )
        
        # # # 简单特征融合 - 直接相加
        # drug_final =  drug_out
        # dis_final = dis_out
        drug_final = drug_out + drug_sim_out 
        dis_final = dis_out + dis_sim_out
        # drug_final = th.cat((drug_out, drug_sim_out), 1)
        # dis_final = th.cat((dis_out, dis_sim_out), 1)

        
        # 应用解码器
        pred_ratings = self.decoder(dec_graph, drug_final, dis_final)
        
        return pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out


# 辅助函数
def udf_u_mul_e_norm(edges):
    return {'reg': edges.src['reg'] * edges.dst['ci']}

def udf_u_mul_e(edges):
    return {'m': th.cat([edges.src['h'], edges.dst['h']], 1)}

def dot_or_identity(A, B, device=None):
    if A is None:
        return B
    elif A.shape[1] == 3:
        if device is None:
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1)
        else:
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1).to(device)
    else:
        return A
