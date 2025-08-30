import csv
import re
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import torch.nn.functional as F

class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()


def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) + \
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt):
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError


def to_etype_name(rating):
    return str(rating).replace('.', '_')


# def common_loss(emb1, emb2):
#     emb1 = emb1 - th.mean(emb1, dim=0, keepdim=True)
#     emb2 = emb2 - th.mean(emb2, dim=0, keepdim=True)
#     emb1 = th.nn.functional.normalize(emb1, p=2, dim=1)
#     emb2 = th.nn.functional.normalize(emb2, p=2, dim=1)
#     cov1 = th.matmul(emb1, emb1.t())
#     cov2 = th.matmul(emb2, emb2.t())
#     cost = th.mean((cov1 - cov2) ** 2)
#     return cost
def projection(args, z: th.Tensor) -> th.Tensor:
    fc1 = th.nn.Linear(args.num_hidden, args.num_proj_hidden1).to(args.device)
    fc2 = th.nn.Linear(args.num_proj_hidden1, args.num_proj_hidden2).to(args.device)
    fc3 = th.nn.Linear(args.num_proj_hidden2, args.num_hidden).to(args.device)
    z1 = F.elu(fc1(z))
    z2 = F.elu(fc2(z1))
    # z = th.sigmoid(fc1(z))
    return fc3(z2)
def sim(z1: th.Tensor, z2: th.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return th.mm(z1, z2.t())

def semi_loss(args, z1: th.Tensor, z2: th.Tensor, flag: int):
    # if flag == 0:
    #     f = lambda x: th.exp(x / args.tau_drug)
    # else:
    #     f = lambda x: th.exp(x / args.tau_disease)
    f = lambda x: th.exp(x / args.tau)
    refl_sim = f(args.intra * sim(z1, z1))  # torch.Size([663, 663])
    between_sim = f(args.inter * sim(z1, z2))  # z1 z2:torch.Size([663, 75])
    # refl_sim = f(sim(z1, z1))  # torch.Size([663, 663])
    # between_sim = f(sim(z1, z2))  # z1 z2:torch.Size([663, 75])
    # refl_sim = (F.cosine_similarity(z1, z1))  # torch.Size([663])
    # between_sim = f(F.cosine_similarity(z1, z2))

    return -th.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def batched_semi_loss(args, z1: th.Tensor, z2: th.Tensor,
                        batch_size: int):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: th.exp(x / args.tau)
    indices = th.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = f(sim(z1[mask], z1))  # [B, N]
        between_sim = f(sim(z1[mask], z2))  # [B, N]

        losses.append(-th.log(
            between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            / (refl_sim.sum(1) + between_sim.sum(1)
                - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

    return th.cat(losses)

def LOSS(args, z1: th.Tensor, z2: th.Tensor,
        mean: bool = True, batch_size: int = 0, flag: int = 0):
    h1 = projection(args, z1)
    h2 = projection(args, z2)

    if batch_size == 0:
        l1 = semi_loss(args, h1, h2, flag)
        l2 = semi_loss(args, h2, h1, flag)
    else:
        l1 = batched_semi_loss(h1, h2, batch_size)
        l2 = batched_semi_loss(h2, h1, batch_size)
    # if batch_size == 0:
    #     l1 = semi_loss(args, z1, z2)
    #     l2 = semi_loss(args, z2, z1)
    # else:
    #     l1 = batched_semi_loss(args, z1, z2, batch_size)
    #     l2 = batched_semi_loss(args, z2, z1, batch_size)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret





import torch as th
import torch.nn.functional as F

def improved_contrastive_loss(args, z1: th.Tensor, z2: th.Tensor, 
                             mean: bool = True, batch_size: int = 0):
    """
    Improved contrastive loss with:
    1. Stable projection with layer normalization
    2. Balanced positive-negative sampling
    3. Adaptive temperature with bounded range
    4. Optional orthogonal regularization
    """
    # Stable projection with layer normalization
    h1 = stable_projection(args, z1)
    h2 = stable_projection(args, z2)
    
    # Compute contrastive loss
    if batch_size == 0:
        l1 = improved_semi_loss(args, h1, h2)
        l2 = improved_semi_loss(args, h2, h1)
    else:
        l1 = batched_improved_semi_loss(args, h1, h2, batch_size)
        l2 = batched_improved_semi_loss(args, h2, h1, batch_size)
    
    # Main contrastive loss
    ret = (l1 + l2) * 0.5
    
    # Optional orthogonal regularization (only if ortho_weight exists)
    if hasattr(args, 'ortho_weight') and args.ortho_weight > 0:
        ortho_loss = orthogonal_regularization(h1, h2)
        ret = ret + args.ortho_weight * ortho_loss
    
    ret = ret.mean() if mean else ret.sum()
    return ret

def stable_projection(args, z: th.Tensor) -> th.Tensor:
    """Projection with layer normalization and residual connection"""
    # Check if projection layers exist, if not create them
    if not hasattr(args, '_proj_layers'):
        args._proj_layers = {}
    
    device = z.device
    key = f"{args.num_hidden}_{args.num_proj_hidden1}_{args.num_proj_hidden2}"
    
    if key not in args._proj_layers:
        fc1 = th.nn.Linear(args.num_hidden, args.num_proj_hidden1).to(device)
        fc2 = th.nn.Linear(args.num_proj_hidden1, args.num_proj_hidden2).to(device)
        fc3 = th.nn.Linear(args.num_proj_hidden2, args.num_hidden).to(device)
        ln1 = th.nn.LayerNorm(args.num_proj_hidden1).to(device)
        ln2 = th.nn.LayerNorm(args.num_proj_hidden2).to(device)
        residual_fc = th.nn.Linear(args.num_hidden, args.num_hidden).to(device)
        
        args._proj_layers[key] = {
            'fc1': fc1, 'fc2': fc2, 'fc3': fc3,
            'ln1': ln1, 'ln2': ln2, 'residual_fc': residual_fc
        }
    
    layers = args._proj_layers[key]
    
    # Main path with layer normalization
    z1 = layers['ln1'](F.elu(layers['fc1'](z)))
    z2 = layers['ln2'](F.elu(layers['fc2'](z1)))
    main_path = layers['fc3'](z2)
    
    # Residual connection
    residual = layers['residual_fc'](z)
    
    # Combine and normalize
    output = main_path + 0.2 * residual
    return F.normalize(output, dim=1)

def improved_semi_loss(args, z1: th.Tensor, z2: th.Tensor):
    """Improved semi loss with balanced sampling and adaptive temperature"""
    # Similarity matrices
    sim_z1z1 = th.mm(z1, z1.t())  # Intra-modal similarity
    sim_z1z2 = th.mm(z1, z2.t())  # Inter-modal similarity
    
    # Adaptive temperature with bounds
    pos_sim = sim_z1z2.diag()
    tau_min = getattr(args, 'tau_min', args.tau * 0.5)
    tau_max = getattr(args, 'tau_max', args.tau * 2.0)
    tau = th.clamp(args.tau * (1.0 + 0.2 * (1.0 - pos_sim.mean())), 
                   min=tau_min, max=tau_max)
    
    # Exponential function
    f = lambda x: th.exp(x / tau)
    
    # Positive pairs
    pos_exp = f(args.inter * pos_sim)
    
    # Negative pairs with masking
    N = z1.size(0)
    neg_mask = 1 - th.eye(N, device=z1.device)
    
    # All negative similarities
    neg_z1z1 = f(args.intra * sim_z1z1) * neg_mask
    neg_z1z2 = f(args.inter * sim_z1z2) * neg_mask
    
    # Hard negative mining: select top-k negatives
    k = max(1, min(N-1, N // 5))  # Top 20% negatives, at least 1
    
    if N > 1:
        topk_neg_z1z1, _ = th.topk(neg_z1z1, k, dim=1)
        topk_neg_z1z2, _ = th.topk(neg_z1z2, k, dim=1)
        neg_sum = topk_neg_z1z1.sum(1) + topk_neg_z1z2.sum(1)
    else:
        neg_sum = th.zeros_like(pos_exp)
    
    # Denominator with numerical stability
    denominator = pos_exp + neg_sum + 1e-8
    
    # Loss
    loss = -th.log(pos_exp / denominator)
    return loss

def batched_improved_semi_loss(args, z1: th.Tensor, z2: th.Tensor, batch_size: int):
    """Batched version of improved semi loss"""
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = th.arange(0, num_nodes).to(device)
    losses = []
    
    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        batch_z1 = z1[mask]
        current_batch_size = len(mask)
        
        # Similarity matrices
        sim_z1z1 = th.mm(batch_z1, z1.t())  # [B, N]
        sim_z1z2 = th.mm(batch_z1, z2.t())  # [B, N]
        
        # Adaptive temperature
        pos_indices = mask
        pos_sim = sim_z1z2[th.arange(current_batch_size), pos_indices]
        tau_min = getattr(args, 'tau_min', args.tau * 0.5)
        tau_max = getattr(args, 'tau_max', args.tau * 2.0)
        tau = th.clamp(args.tau * (1.0 + 0.2 * (1.0 - pos_sim.mean())), 
                       min=tau_min, max=tau_max)
        
        # Exponential function
        f = lambda x: th.exp(x / tau)
        
        # Positive pairs
        pos_exp = f(args.inter * pos_sim)
        
        # Negative mask for current batch
        batch_neg_mask = th.ones_like(sim_z1z1)
        batch_neg_mask[th.arange(current_batch_size), pos_indices] = 0
        
        # Negative similarities
        neg_z1z1 = f(args.intra * sim_z1z1) * batch_neg_mask
        neg_z1z2 = f(args.inter * sim_z1z2) * batch_neg_mask
        
        # Hard negative mining
        k = max(1, min(num_nodes-1, num_nodes // 5))
        
        if num_nodes > 1:
            topk_neg_z1z1, _ = th.topk(neg_z1z1, k, dim=1)
            topk_neg_z1z2, _ = th.topk(neg_z1z2, k, dim=1)
            neg_sum = topk_neg_z1z1.sum(1) + topk_neg_z1z2.sum(1)
        else:
            neg_sum = th.zeros_like(pos_exp)
        
        # Loss
        denominator = pos_exp + neg_sum + 1e-8
        batch_loss = -th.log(pos_exp / denominator)
        losses.append(batch_loss)
    
    return th.cat(losses)

def orthogonal_regularization(h1: th.Tensor, h2: th.Tensor) -> th.Tensor:
    """Encourage orthogonal representations"""
    device = h1.device
    d = h1.size(1)
    
    # Covariance matrices
    cov_h1 = th.mm(h1.t(), h1) / h1.size(0)
    cov_h2 = th.mm(h2.t(), h2) / h2.size(0)
    
    # Identity matrix
    identity = th.eye(d, device=device)
    
    # Orthogonal loss
    ortho_loss = (th.norm(cov_h1 - identity, 'fro') ** 2 + 
                  th.norm(cov_h2 - identity, 'fro') ** 2) / (2 * d)
    
    return ortho_loss




#sy
import torch as th
import torch.nn.functional as F

def infonce_loss_builtin(z1: th.Tensor, z2: th.Tensor, tau: float = 0.5):
    """
    使用PyTorch内置函数实现InfoNCE
    """
    N = z1.size(0)
    device = z1.device
    
    # 计算相似度矩阵
    sim_matrix = th.mm(z1, z2.t()) / tau  # [N, N]
    
    # 标签：每个样本的正样本就是它自己对应的位置
    labels = th.arange(N, device=device)
    
    # 使用交叉熵损失
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss

def LOSS1(args, z1: th.Tensor, z2: th.Tensor):
    """
    双向 InfoNCE 损失
    参数:
        args : 超参数 Namespace，需包含 tau
        z1   : [N, D] 张量
        z2   : [N, D] 张量
    """
    tau = 0.5         # 从 args 里取温度系数
    # 归一化
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # 双向损失
    loss1 = infonce_loss_builtin(z1, z2, tau)
    loss2 = infonce_loss_builtin(z2, z1, tau)

    return (loss1 + loss2) / 2


import torch as th
import torch.nn.functional as F

def graph_triplet_loss(z1: th.Tensor, z2: th.Tensor, z_neg: th.Tensor = None, 
                      margin: float = 1.0, mining_strategy: str = 'hard'):
    """
    图对比学习中的Triplet Loss
    Args:
        z1: [N, D] - anchor表示
        z2: [N, D] - positive表示（同一节点的另一个视图）
        z_neg: [N, D] - negative表示（可选，如果没有则随机采样）
        margin: triplet loss的margin
        mining_strategy: 'hard', 'random', 'semi_hard'
    """
    N = z1.size(0)
    device = z1.device
    
    # 归一化
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # 如果没有提供负样本，则从正样本中采样
    if z_neg is None:
        z_neg = mine_negatives(z1, z2, strategy=mining_strategy)
    else:
        z_neg = F.normalize(z_neg, dim=1)
    
    # 使用PyTorch官方的triplet loss
    triplet_loss = th.nn.TripletMarginLoss(margin=margin, p=2)
    loss = triplet_loss(z1, z2, z_neg)
    
    return loss

def mine_negatives(z1: th.Tensor, z2: th.Tensor, strategy: str = 'hard'):
    """
    负样本挖掘策略
    """
    N = z1.size(0)
    device = z1.device
    
    if strategy == 'random':
        # 随机负采样
        perm = th.randperm(N, device=device)
        return z2[perm]
    
    elif strategy == 'hard':
        # 硬负采样：选择与anchor最相似的负样本
        sim_matrix = th.mm(z1, z2.t())
        
        # 排除对角线（正样本）
        mask = ~th.eye(N, device=device).bool()
        neg_indices = []
        
        for i in range(N):
            # 找到与anchor i最相似的负样本
            candidates = sim_matrix[i][mask[i]]
            if len(candidates) > 0:
                hard_neg_idx = th.argmax(candidates)
                # 转换回原始索引
                available_indices = th.arange(N, device=device)[mask[i]]
                neg_indices.append(available_indices[hard_neg_idx])
            else:
                neg_indices.append((i + 1) % N)  # fallback
        
        return z2[th.tensor(neg_indices, device=device)]
    
    elif strategy == 'semi_hard':
        # 半硬负采样：选择距离在[d_pos, d_pos + margin]之间的负样本
        pos_dist = F.pairwise_distance(z1, z2)
        sim_matrix = th.mm(z1, z2.t())
        
        neg_indices = []
        for i in range(N):
            pos_d = pos_dist[i]
            candidates = []
            
            for j in range(N):
                if i != j:
                    neg_d = 1 - sim_matrix[i, j]  # 转换为距离
                    if pos_d < neg_d < pos_d + 1.0:  # semi-hard条件
                        candidates.append(j)
            
            if candidates:
                neg_indices.append(th.tensor(candidates, device=device)[0])
            else:
                neg_indices.append((i + 1) % N)  # fallback
        
        return z2[th.tensor(neg_indices, device=device)]


def LOSS2(args, z1: th.Tensor, z2: th.Tensor, 
                                 mean: bool = True, batch_size: int = 0):
        # 投影
        h1 = projection_head(args, z1)
        h2 = projection_head(args, z2)

        margin = getattr(args, 'margin', 1.0)           # 取不到就用 1.0
        mining_strategy = getattr(args, 'mining_strategy', 'hard')

        # 使用triplet loss
        loss = graph_triplet_loss(h1, h2, margin=margin, 
                                mining_strategy=mining_strategy)
        
        return loss


def projection_head(args, z):
    """投影头"""
    if not hasattr(args, '_proj'):
        args._proj = th.nn.Sequential(
            th.nn.Linear(args.num_hidden, args.num_proj_hidden1),
            th.nn.ReLU(),
            th.nn.Linear(args.num_proj_hidden1, args.num_proj_hidden2)
        ).to(z.device)
    
    return F.normalize(args._proj(z), dim=1)
