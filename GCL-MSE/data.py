import os
import dgl
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch as th
import torch.nn.functional as F
import torch 
import numpy as np
import torch.optim as optim
from utils import *
from sklearn.model_selection import KFold


_paths = {
    'Gdataset': './DRGCL-main/DRGCL-main/raw_data/drug_data/Gdataset/Gdataset.mat',
    'twoGdataset': './DRGCL-main/DRGCL-main/raw_data/drug_data/Gdataset/819twoGdataset.mat',
    'Cdataset': './DRGCL-main/DRGCL-main/raw_data/drug_data/Cdataset/Cdataset.mat',
    'twoCdataset': './DRGCL-main/DRGCL-main/raw_data/drug_data/Cdataset/twoCdataset.mat',
    'lrssl': './DRGCL-main/DRGCL-main/raw_data/drug_data/lrssl',
    'twolrssl': './DRGCL-main/DRGCL-main/raw_data/drug_data/lrssl/twolrssl.mat',


}


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


class DrugNovoLoader(object):
    def __init__(self,
                 name,
                 device,
                 symm=True,
                 k=2):
        self._name = name
        self._device = device
        self._symm = symm
        self.num_neighbor = k
        print("Starting processing {} ...".format(self._name))
        self._dir = os.path.join(_paths[self._name])
        self.cv_data_dict = self._load_drug_data(self._dir, self._name)

        self._generate_topoy_graph()
        self.drug_graph, self.disease_graph = self._generate_feat_graph()
        self._generate_feat()
        self.possible_rel_values = self.values

    def _load_drug_data(self, file_path, data_name):
        association_matrix = None
        if data_name in ['Gdataset', 'Cdataset']:
            data = sio.loadmat(file_path)
            association_matrix = data['didr'].T
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']
        elif data_name in ['Ldataset']:
            association_matrix = np.loadtxt(os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
            self.disease_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
            self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            association_matrix = data.values
            self.disease_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values

        # self.disease_sim_features = th.FloatTensor(self.disease_sim_features)
        # self.drug_sim_features = th.FloatTensor(self.drug_sim_features)


        # row_num_count = association_matrix.sum(axis=1)  # 求每行之和
        # self.row_idx = np.where(row_num_count == 1)[0]

        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]
        self.row_idx = th.arange(0,self._num_drug).numpy()

        cv_num = 0
        cv_data = {}
        for idx in self.row_idx:
            association_matrix1 = association_matrix.copy()
            test_value = association_matrix1[idx, :]
            test_data = {
                'drug_idx': [idx] * len(test_value),
                'disease_idx': [col for col in range(0, self._num_disease)],
                'values': test_value
            }
            test_data_info = pd.DataFrame(test_data, index=None)

            association_matrix1[idx, :] = 0
            pos_row, pos_col = np.nonzero(association_matrix1)
            neg_row, neg_col = np.nonzero(1 - association_matrix1)

            train_drug_idx = np.hstack([pos_row, neg_row])
            train_disease_idx = np.hstack([pos_col, neg_col])

            pos_values = [1] * len(pos_row)
            neg_values = [0] * len(neg_row)
            train_values = np.hstack([pos_values, neg_values])

            train_data = {
                'drug_idx': train_drug_idx,
                'disease_idx': train_disease_idx,
                'values': train_values
            }
            train_data_info = pd.DataFrame(train_data, index=None)

            values = np.unique(train_values)
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data


    def _generate_feat(self):
        self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3)
        self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3)

        self.drug_feature = th.cat(
            [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
             th.zeros([self.num_drug, 1])], 1)

        self.disease_feature = th.cat(
            [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
             th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1)

    def _generate_topoy_graph(self):
        self.data_cv = {}
        for cv in range(0, len(self.row_idx)):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values

            train_pairs, train_values = self._generate_pair_value(self.train_rel_info)

            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)

            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv
    def _generate_feat_graph(self):
        # drug feature graph
        drug_sim = self.drug_sim_features
        drug_num_neighbor = self.num_neighbor
        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]

        drug_neighbor = np.argpartition(-drug_sim, kth=drug_num_neighbor, axis=1)[:, :drug_num_neighbor]
        dr_row_index = np.arange(drug_neighbor.shape[0]).repeat(drug_neighbor.shape[1])
        dr_col_index = drug_neighbor.reshape(-1)
        drug_edge_index = np.array([dr_row_index, dr_col_index]).astype(int).T

        drug_edges = np.array(list(drug_edge_index), dtype=np.int32).reshape(drug_edge_index.shape)
        drug_adj = sp.coo_matrix((np.ones(drug_edges.shape[0]), (drug_edges[:, 0], drug_edges[:, 1])),
                                 shape=(self.num_drug, self.num_drug),
                                 dtype=np.float32)
        drug_adj = drug_adj + drug_adj.T.multiply(drug_adj.T > drug_adj) - drug_adj.multiply(
            drug_adj.T > drug_adj)
        # drug_graph = normalize(drug_adj + sp.eye(drug_adj.shape[0]))
        drug_graph = normalize(drug_adj)
        drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)
        # disease feature graph
        disease_sim = self.disease_sim_features
        disease_num_neighbor = self.num_neighbor
        if disease_num_neighbor > disease_sim.shape[0] or disease_num_neighbor < 0:
            disease_num_neighbor = disease_sim.shape[0]

        disease_neighbor = np.argpartition(-disease_sim, kth=disease_num_neighbor, axis=1)[:, :disease_num_neighbor]
        di_row_index = np.arange(disease_neighbor.shape[0]).repeat(disease_neighbor.shape[1])
        di_col_index = disease_neighbor.reshape(-1)
        disease_edge_index = np.array([di_row_index, di_col_index]).astype(int).T

        disease_edges = np.array(list(disease_edge_index), dtype=np.int32).reshape(disease_edge_index.shape)
        disease_adj = sp.coo_matrix((np.ones(disease_edges.shape[0]), (disease_edges[:, 0], disease_edges[:, 1])),
                                    shape=(self.num_disease, self.num_disease),
                                    dtype=np.float32)
        disease_adj = disease_adj + disease_adj.T.multiply(disease_adj.T > disease_adj) - disease_adj.multiply(
            disease_adj.T > disease_adj)
        # disease_graph = normalize(disease_adj + sp.eye(disease_adj.shape[0]))
        disease_graph = normalize(disease_adj)
        disease_graph = sparse_mx_to_torch_sparse_tensor(disease_graph)

        return drug_graph, disease_graph

    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["drug_idx"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["disease_idx"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rel_values:
            ridx = np.where(
                rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('drug', str(rating), 'disease'): (rrow, rcol),
                ('disease', 'rev-%s' % str(rating), 'drug'): (rcol, rrow)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            drug_ci = []
            drug_cj = []
            disease_ci = []
            disease_cj = []
            for r in self.possible_rel_values:
                r = to_etype_name(r)
                drug_ci.append(graph['rev-%s' % r].in_degrees())
                disease_ci.append(graph[r].in_degrees())
                if self._symm:
                    drug_cj.append(graph[r].out_degrees())
                    disease_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    drug_cj.append(th.zeros((self.num_drug,)))
                    disease_cj.append(th.zeros((self.num_disease,)))

            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self.num_drug, )
                disease_cj = th.ones(self.num_disease, )
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                     vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug



class DiseaseNovoLoader(object):
    def __init__(self,
                 name,
                 device,
                 symm=True,
                 k=2):
        self._name = name
        self._device = device
        self._symm = symm
        self.num_neighbor = k
        print("Starting processing {} ...".format(self._name))
        self._dir = os.path.join(_paths[self._name])
        self.cv_data_dict = self._load_drug_data(self._dir, self._name)

        self._generate_topoy_graph()
        self.drug_graph, self.disease_graph = self._generate_feat_graph()
        self._generate_feat()
        self.possible_rel_values = self.values

    def _load_drug_data(self, file_path, data_name):
        association_matrix = None
        if data_name in ['Gdataset', 'Cdataset']:
            data = sio.loadmat(file_path)
            association_matrix = data['didr'].T
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']
        elif data_name in ['Ldataset']:
            association_matrix = np.loadtxt(os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
            self.disease_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
            self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            association_matrix = data.values
            self.disease_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values

        # self.disease_sim_features = th.FloatTensor(self.disease_sim_features)
        # self.drug_sim_features = th.FloatTensor(self.drug_sim_features)


        # row_num_count = association_matrix.sum(axis=1)  # 求每行之和
        # self.row_idx = np.where(row_num_count == 1)[0]

        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]
        # self.row_idx = th.arange(0,self._num_drug).numpy()
        self.col_idx = th.arange(0, self._num_disease).numpy()

        cv_num = 0
        cv_data = {}
        for idx in self.col_idx:
            association_matrix1 = association_matrix.copy()
            test_value = association_matrix1[:, idx]
            test_data = {
                'drug_idx': [row for row in range(0, self._num_drug)],
                'disease_idx': [idx] * len(test_value),
                'values': test_value
            }
            test_data_info = pd.DataFrame(test_data, index=None)

            association_matrix1[:, idx] = 0
            pos_row, pos_col = np.nonzero(association_matrix1)
            neg_row, neg_col = np.nonzero(1 - association_matrix1)

            train_drug_idx = np.hstack([pos_row, neg_row])
            train_disease_idx = np.hstack([pos_col, neg_col])

            pos_values = [1] * len(pos_row)
            neg_values = [0] * len(neg_row)
            train_values = np.hstack([pos_values, neg_values])

            train_data = {
                'drug_idx': train_drug_idx,
                'disease_idx': train_disease_idx,
                'values': train_values
            }
            train_data_info = pd.DataFrame(train_data, index=None)

            values = np.unique(train_values)
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data

    def _generate_feat(self):
        self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3)
        self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3)

        self.drug_feature = th.cat(
            [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
             th.zeros([self.num_drug, 1])], 1)

        self.disease_feature = th.cat(
            [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
             th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1)

    def _generate_topoy_graph(self):
        self.data_cv = {}
        for cv in range(0, len(self.col_idx)):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values

            train_pairs, train_values = self._generate_pair_value(self.train_rel_info)

            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)

            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv
    def _generate_feat_graph(self):
        # drug feature graph
        drug_sim = self.drug_sim_features
        drug_num_neighbor = self.num_neighbor
        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]

        drug_neighbor = np.argpartition(-drug_sim, kth=drug_num_neighbor, axis=1)[:, :drug_num_neighbor]
        dr_row_index = np.arange(drug_neighbor.shape[0]).repeat(drug_neighbor.shape[1])
        dr_col_index = drug_neighbor.reshape(-1)
        drug_edge_index = np.array([dr_row_index, dr_col_index]).astype(int).T

        drug_edges = np.array(list(drug_edge_index), dtype=np.int32).reshape(drug_edge_index.shape)
        drug_adj = sp.coo_matrix((np.ones(drug_edges.shape[0]), (drug_edges[:, 0], drug_edges[:, 1])),
                                 shape=(self.num_drug, self.num_drug),
                                 dtype=np.float32)
        drug_adj = drug_adj + drug_adj.T.multiply(drug_adj.T > drug_adj) - drug_adj.multiply(
            drug_adj.T > drug_adj)
        # drug_graph = normalize(drug_adj + sp.eye(drug_adj.shape[0]))
        drug_graph = normalize(drug_adj)
        drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)
        # disease feature graph
        disease_sim = self.disease_sim_features
        disease_num_neighbor = self.num_neighbor
        if disease_num_neighbor > disease_sim.shape[0] or disease_num_neighbor < 0:
            disease_num_neighbor = disease_sim.shape[0]

        disease_neighbor = np.argpartition(-disease_sim, kth=disease_num_neighbor, axis=1)[:, :disease_num_neighbor]
        di_row_index = np.arange(disease_neighbor.shape[0]).repeat(disease_neighbor.shape[1])
        di_col_index = disease_neighbor.reshape(-1)
        disease_edge_index = np.array([di_row_index, di_col_index]).astype(int).T

        disease_edges = np.array(list(disease_edge_index), dtype=np.int32).reshape(disease_edge_index.shape)
        disease_adj = sp.coo_matrix((np.ones(disease_edges.shape[0]), (disease_edges[:, 0], disease_edges[:, 1])),
                                    shape=(self.num_disease, self.num_disease),
                                    dtype=np.float32)
        disease_adj = disease_adj + disease_adj.T.multiply(disease_adj.T > disease_adj) - disease_adj.multiply(
            disease_adj.T > disease_adj)
        # disease_graph = normalize(disease_adj + sp.eye(disease_adj.shape[0]))
        disease_graph = normalize(disease_adj)
        disease_graph = sparse_mx_to_torch_sparse_tensor(disease_graph)

        return drug_graph, disease_graph

    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["drug_idx"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["disease_idx"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rel_values:
            ridx = np.where(
                rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('drug', str(rating), 'disease'): (rrow, rcol),
                ('disease', 'rev-%s' % str(rating), 'drug'): (rcol, rrow)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            drug_ci = []
            drug_cj = []
            disease_ci = []
            disease_cj = []
            for r in self.possible_rel_values:
                r = to_etype_name(r)
                drug_ci.append(graph['rev-%s' % r].in_degrees())
                disease_ci.append(graph[r].in_degrees())
                if self._symm:
                    drug_cj.append(graph[r].out_degrees())
                    disease_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    drug_cj.append(th.zeros((self.num_drug,)))
                    disease_cj.append(th.zeros((self.num_disease,)))

            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self.num_drug, )
                disease_cj = th.ones(self.num_disease, )
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                     vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug
    
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import dgl.function as fn


class ImprovedGATLayer(nn.Module): 
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1): 
        super(ImprovedGATLayer, self).__init__() 
        self.num_heads = num_heads 
        self.out_features = out_features
        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False) 
            for _ in range(num_heads)
        ])
        self.a = nn.ModuleList([
            nn.Linear(2 * out_features, 1, bias=False)
            for _ in range(num_heads)
        ])
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_proj = nn.Linear(num_heads * out_features, out_features)
        
        # 稳定初始化
        for i in range(num_heads):
            nn.init.xavier_normal_(self.W[i].weight, gain=0.5)
            nn.init.xavier_normal_(self.a[i].weight, gain=0.5)
        nn.init.xavier_normal_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.out_proj.bias)

    def edge_attention(self, edges, head_idx):
        # 确保特征不包含NaN
        src_feat = torch.where(torch.isnan(edges.src['h']), 
                              torch.zeros_like(edges.src['h']), 
                              edges.src['h'])
        dst_feat = torch.where(torch.isnan(edges.dst['h']), 
                              torch.zeros_like(edges.dst['h']), 
                              edges.dst['h'])
        
        # 限制特征范围
        src_feat = torch.clamp(src_feat, -10, 10)
        dst_feat = torch.clamp(dst_feat, -10, 10)
        
        feat = torch.cat([src_feat, dst_feat], dim=1)
        attention = self.a[head_idx](feat)
        attention = self.leaky_relu(attention)
        
        # 数值稳定：避免exp爆炸
        return {'alpha': torch.exp(torch.clamp(attention, -5, 5))}
    
    def forward(self, g, features):
        # 检查输入特征
        features = torch.where(torch.isnan(features), 
                              torch.zeros_like(features), 
                              features)
        features = torch.clamp(features, -10, 10)
        
        h_heads = []
        for i in range(self.num_heads):
            h = self.W[i](features)
            
            with g.local_scope():
                g.ndata['h'] = h
                g.apply_edges(lambda edges: self.edge_attention(edges, i))
                g.edata['alpha'] = self.dropout(g.edata['alpha'])
                g.update_all(
                    fn.u_mul_e('h', 'alpha', 'm'),
                    fn.sum('m', 'h_agg')
                )
                h_head = g.ndata['h_agg']
                h_heads.append(h_head)
        
        h_multi = torch.cat(h_heads, dim=1)
        out = self.out_proj(h_multi)
        return F.elu(out)



def edge_attention(self, edges, head_idx):
    h_cat = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
    e = self.leaky_relu(self.a[head_idx](h_cat))
    return {'alpha': e}


class AdaptiveKPredictor(nn.Module): 
    def __init__(self, in_features, hidden_dim=128): 
        super(AdaptiveKPredictor, self).__init__() 
        self.hidden_dim = hidden_dim
        self.in_features = in_features
        self.initialized = False
        
    def _initialize(self, input_dim):
        """更复杂的网络架构"""
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2) 
        self.bn2 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim // 4)
        self.fc4 = nn.Linear(self.hidden_dim // 4, 1)
        self.dropout = nn.Dropout(0.3)
        
        # 不同的初始化以产生多样化输出
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.8)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.8)
        nn.init.normal_(self.fc4.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc4.bias, 0.5)  # 偏置初始化为中间值
        
        self.initialized = True

    def forward(self, node_features, graph_stats):
        # 处理输入
        node_features = torch.where(torch.isnan(node_features), 
                                  torch.zeros_like(node_features), 
                                  node_features)
        graph_stats = torch.where(torch.isnan(graph_stats), 
                                torch.zeros_like(graph_stats), 
                                graph_stats)
        
        # 添加噪声鼓励多样性
        if self.training:
            noise = torch.randn_like(node_features) * 0.01
            node_features = node_features + noise
        
        x = torch.cat([node_features, graph_stats], dim=1)
        
        if not self.initialized:
            input_dim = x.size(1)
            self._initialize(input_dim)
            self.to(x.device)
        
        x = torch.clamp(x, -10, 10)
        
        batch_size = x.size(0)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        # 使用tanh而非sigmoid以产生更多样的输出
        x = 0.5 + 0.4 * torch.tanh(self.fc4(x))
        
        return x







class NewDrugDataLoader(object):
    def __init__(self,
             name,
             device,
             symm=True,
             learning_rate=0.001,
             k_min=1,
             k_max=20,
             attention_epochs=100):
        
        self._name = name
        self._device = device
        self._symm = symm
        # self.base_k = base_k
        # self.k_strategy = k_strategy
        self.learning_rate = learning_rate
        self.k_min = k_min
        self.k_max = k_max
        # self.k_expand_factor = k_expand_factor

        
        print("Starting processing {} ...".format(self._name))
        self._dir = os.path.join(_paths[self._name])
        self.cv_data_dict = self._load_drug_data(self._dir, self._name)
        
        # 特征预处理
        self._preprocess_features()
        
        # 初始化改进的GAT层
        self.drug_gat = ImprovedGATLayer(
            self.drug_feature_dim, 
            self.drug_feature_dim, 
            num_heads=4
        ).to(device)
        
        self.disease_gat = ImprovedGATLayer(
            self.disease_feature_dim,
            self.disease_feature_dim,
            num_heads=4
        ).to(device)
        
        # 初始化K预测器
        self.drug_k_predictor = AdaptiveKPredictor(
            self.drug_feature_dim,
        ).to(device)
        
        self.disease_k_predictor = AdaptiveKPredictor(
            self.disease_feature_dim,
        ).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.drug_gat.parameters(), 'lr': self.learning_rate},
            {'params': self.disease_gat.parameters(), 'lr': self.learning_rate},
            {'params': self.drug_k_predictor.parameters(), 'lr': self.learning_rate * 0.1},
            {'params': self.disease_k_predictor.parameters(), 'lr': self.learning_rate * 0.1}
        ], weight_decay=1e-5)  # 添加权重衰减
        
        # 训练模型
        self._train_models(epochs=attention_epochs)
        
        # 生成图
        self._generate_topoy_graph()
        self.drug_graph, self.disease_graph = self._generate_feat_graph()
        self._generate_feat()
        self.possible_rel_values = self.values

    def _preprocess_features(self):
        """特征预处理和增强"""
        # 标准化相似度矩阵
        scaler = StandardScaler()
        
        # 药物特征增强
        drug_sim_flat = self.drug_sim_features.reshape(-1, 1)
        drug_sim_normalized = scaler.fit_transform(drug_sim_flat).reshape(self.drug_sim_features.shape)
        
        # 计算额外的统计特征
        drug_degree = np.sum(self.drug_sim_features > 0.5, axis=1, keepdims=True)
        drug_avg_sim = np.mean(self.drug_sim_features, axis=1, keepdims=True)
        drug_std_sim = np.std(self.drug_sim_features, axis=1, keepdims=True)
        
        # 拼接特征
        self.drug_enhanced_features = np.concatenate([
            drug_sim_normalized,
            drug_degree / self.num_drug,  # 归一化度
            drug_avg_sim,
            drug_std_sim
        ], axis=1)
        
        # 疾病特征增强
        disease_sim_flat = self.disease_sim_features.reshape(-1, 1)
        disease_sim_normalized = scaler.fit_transform(disease_sim_flat).reshape(self.disease_sim_features.shape)
        
        disease_degree = np.sum(self.disease_sim_features > 0.5, axis=1, keepdims=True)
        disease_avg_sim = np.mean(self.disease_sim_features, axis=1, keepdims=True)
        disease_std_sim = np.std(self.disease_sim_features, axis=1, keepdims=True)
        
        self.disease_enhanced_features = np.concatenate([
            disease_sim_normalized,
            disease_degree / self.num_disease,
            disease_avg_sim,
            disease_std_sim
        ], axis=1)
        
        self.drug_feature_dim = self.drug_enhanced_features.shape[1]
        self.disease_feature_dim = self.disease_enhanced_features.shape[1]

    def _compute_graph_statistics(self, features):
        """计算图的全局统计信息"""
        stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'max': np.max(features, axis=0),
            'min': np.min(features, axis=0)
        }
        # 广播到每个节点
        graph_stats = np.concatenate([
            np.tile(stats['mean'], (features.shape[0], 1)),
            np.tile(stats['std'], (features.shape[0], 1)),
        ], axis=1)
        return graph_stats

    def _train_models(self, epochs=100):
            """改进的训练过程，让K值自然反映节点特性"""
            print("Training attention mechanism and K predictors...")
            
            # 准备训练数据
            drug_features = torch.FloatTensor(self.drug_enhanced_features).to(self._device)
            disease_features = torch.FloatTensor(self.disease_enhanced_features).to(self._device)
            
            # 计算图统计信息
            drug_stats = torch.FloatTensor(
                self._compute_graph_statistics(self.drug_enhanced_features)
            ).to(self._device)
            disease_stats = torch.FloatTensor(
                self._compute_graph_statistics(self.disease_enhanced_features)
            ).to(self._device)
            
            # 检查并替换NaN值
            drug_features = torch.where(torch.isnan(drug_features), 
                                    torch.zeros_like(drug_features), 
                                    drug_features)
            disease_features = torch.where(torch.isnan(disease_features), 
                                        torch.zeros_like(disease_features), 
                                        disease_features)
            drug_stats = torch.where(torch.isnan(drug_stats), 
                                torch.zeros_like(drug_stats), 
                                drug_stats)
            disease_stats = torch.where(torch.isnan(disease_stats), 
                                    torch.zeros_like(disease_stats), 
                                    disease_stats)
            
            # 创建初始图（使用较小的固定K值）
            init_k = min(5, self.num_drug - 1)
            drug_graph = self._create_knn_graph(self.drug_sim_features, init_k, self.num_drug)
            disease_graph = self._create_knn_graph(self.disease_sim_features, init_k, self.num_disease)
            
            best_loss = float('inf')
            patience = 20
            patience_counter = 0
            epsilon = 1e-8  # 用于数值稳定性
            
            # 使用学习率调度器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                
                # 前向传播 - GAT
                drug_embeddings = self.drug_gat(drug_graph, drug_features)
                disease_embeddings = self.disease_gat(disease_graph, disease_features)
                
                # 检查嵌入是否包含NaN
                if torch.isnan(drug_embeddings).any() or torch.isnan(disease_embeddings).any():
                    print("Warning: NaN embeddings detected")
                    drug_embeddings = torch.where(torch.isnan(drug_embeddings), 
                                                torch.zeros_like(drug_embeddings), 
                                                drug_embeddings)
                    disease_embeddings = torch.where(torch.isnan(disease_embeddings), 
                                                torch.zeros_like(disease_embeddings), 
                                                disease_embeddings)
                
                # 计算重建损失
                drug_sim_pred = torch.mm(drug_embeddings, drug_embeddings.t())
                disease_sim_pred = torch.mm(disease_embeddings, disease_embeddings.t())
                
                # 使用加权损失，更重视高相似度的边
                drug_sim_true = torch.FloatTensor(self.drug_sim_features).to(self._device)
                disease_sim_true = torch.FloatTensor(self.disease_sim_features).to(self._device)
                
                # 加权MSE损失
                drug_weights = 1 + drug_sim_true * 2  # 高相似度边权重更大
                disease_weights = 1 + disease_sim_true * 2
                
                drug_recon_loss = torch.mean(drug_weights * (drug_sim_pred - drug_sim_true) ** 2 + epsilon)
                disease_recon_loss = torch.mean(disease_weights * (disease_sim_pred - disease_sim_true) ** 2 + epsilon)
                
                # K预测损失
                drug_k_pred = self.drug_k_predictor(drug_embeddings, drug_stats)
                disease_k_pred = self.disease_k_predictor(disease_embeddings, disease_stats)
                
                # 检查预测值是否包含NaN
                if torch.isnan(drug_k_pred).any() or torch.isnan(disease_k_pred).any():
                    print("Warning: NaN K predictions detected")
                    drug_k_pred = torch.where(torch.isnan(drug_k_pred), 
                                            torch.ones_like(drug_k_pred) * 0.5, 
                                            drug_k_pred)
                    disease_k_pred = torch.where(torch.isnan(disease_k_pred), 
                                                torch.ones_like(disease_k_pred) * 0.5, 
                                                disease_k_pred)
                
                # K值的目标：基于节点的度和相似度分布
                drug_k_target = self._compute_k_targets(self.drug_sim_features, self.k_min, self.k_max)
                disease_k_target = self._compute_k_targets(self.disease_sim_features, self.k_min, self.k_max)
                
                drug_k_target = torch.FloatTensor(drug_k_target).unsqueeze(1).to(self._device)
                disease_k_target = torch.FloatTensor(disease_k_target).unsqueeze(1).to(self._device)
                
                # K预测损失 - 不添加多样性惩罚
                k_loss = F.mse_loss(drug_k_pred, drug_k_target) + F.mse_loss(disease_k_pred, disease_k_target)
                
                # 输出当前预测的K值分布
                if epoch % 10 == 0:
                    drug_k_mean = torch.mean(drug_k_pred).item()
                    drug_k_std = torch.std(drug_k_pred).item()
                    disease_k_mean = torch.mean(disease_k_pred).item()
                    disease_k_std = torch.std(disease_k_pred).item()
                    print(f"  K predictions - Drug: mean={drug_k_mean:.3f}, std={drug_k_std:.3f} | "
                        f"Disease: mean={disease_k_mean:.3f}, std={disease_k_std:.3f}")
                
                # 总损失
                total_loss = drug_recon_loss + disease_recon_loss + 0.1 * k_loss + epsilon
                
                # 检查损失是否为NaN
                if torch.isnan(total_loss).any():
                    print(f"Warning: NaN loss at epoch {epoch+1}")
                    # 使用上一次的有效损失或设置默认损失
                    total_loss = torch.tensor(1.0, requires_grad=True, device=self._device)
                
                # 添加正则化
                l2_reg = 0.0001 * sum(p.pow(2).sum() for p in self.drug_gat.parameters())
                total_loss += l2_reg
                
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    list(self.drug_gat.parameters()) + 
                    list(self.disease_gat.parameters()) +
                    list(self.drug_k_predictor.parameters()) +
                    list(self.disease_k_predictor.parameters()), 
                    max_norm=0.5
                )

                self.optimizer.step()
                
                # 更新学习率调度器
                scheduler.step(total_loss)
                
                # 每15轮更新一次图结构
                if (epoch + 1) % 15 == 0:
                    with torch.no_grad():
                        # 使用当前预测的K值更新图
                        drug_k_values = drug_k_pred.cpu().numpy().flatten()
                        disease_k_values = disease_k_pred.cpu().numpy().flatten()
                        
                        # 将0-1范围的预测值转换为实际K值，不添加随机性
                        drug_k_int = np.clip(
                            np.round(drug_k_values * (self.k_max - self.k_min) + self.k_min),
                            self.k_min, 
                            min(self.k_max, self.num_drug - 1)
                        ).astype(int)
                        
                        disease_k_int = np.clip(
                            np.round(disease_k_values * (self.k_max - self.k_min) + self.k_min),
                            self.k_min,
                            min(self.k_max, self.num_disease - 1)
                        ).astype(int)
                        
                        # 检查K值是否有效
                        if np.any(drug_k_int <= 0) or np.any(disease_k_int <= 0):
                            print("Warning: Invalid K values detected")
                            drug_k_int = np.maximum(drug_k_int, 1)
                            disease_k_int = np.maximum(disease_k_int, 1)
                        
                        # 使用动态K值创建新图
                        drug_graph = self._create_dynamic_knn_graph(
                            self.drug_sim_features, drug_k_int, self.num_drug
                        )
                        disease_graph = self._create_dynamic_knn_graph(
                            self.disease_sim_features, disease_k_int, self.num_disease
                        )
                
                # 早停
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.4f}, "
                        f"Recon: {(drug_recon_loss + disease_recon_loss).item():.4f}, "
                        f"K-Loss: {k_loss.item():.4f}")
            
            # 设置为评估模式
            self.drug_gat.eval()
            self.disease_gat.eval()
            self.drug_k_predictor.eval()
            self.disease_k_predictor.eval()
            
            print("Training completed.")




    def _compute_k_targets(self, sim_matrix, k_min, k_max):
        """基于节点特性计算目标K值，不添加随机性"""
        n = sim_matrix.shape[0]
        k_targets = []
        
        for i in range(n):
            # 分析节点i的连接模式
            sims = sim_matrix[i]
            sorted_sims = np.sort(sims)[::-1]
            
            # 找到相似度的"拐点" - 相似度开始显著下降的位置
            sim_diffs = np.diff(sorted_sims)
            significant_drops = np.where(sim_diffs < -0.1)[0]
            
            if len(significant_drops) > 0:
                # 如果存在明显的相似度下降点，使用它作为K值的基础
                elbow_point = significant_drops[0] + 1
                target_k = (elbow_point - k_min) / (k_max - k_min)
            else:
                # 否则基于相似度分布的整体特性
                high_sim_ratio = np.sum(sims > 0.7) / n
                med_sim_ratio = np.sum((sims > 0.4) & (sims <= 0.7)) / n
                
                if high_sim_ratio > 0.2:
                    target_k = 0.7 + high_sim_ratio * 0.3
                elif med_sim_ratio > 0.3:
                    target_k = 0.4 + med_sim_ratio * 0.3
                else:
                    target_k = 0.3
            
            k_targets.append(np.clip(target_k, 0.1, 0.9))
        
        return np.array(k_targets)



    def _create_knn_graph(self, sim_matrix, k, num_nodes):
        """创建固定K的KNN图"""
        edges = []
        for i in range(num_nodes):
            # 获取top-k邻居
            if k < num_nodes:
                neighbors = np.argpartition(-sim_matrix[i], kth=k)[:k]
            else:
                neighbors = np.arange(num_nodes)
            
            for j in neighbors:
                if i != j:
                    edges.append((i, j))
        
        if edges:
            src, dst = zip(*edges)
            g = dgl.graph((src, dst), num_nodes=num_nodes).to(self._device)
        else:
            g = dgl.graph(([], []), num_nodes=num_nodes).to(self._device)
        
        return dgl.add_self_loop(g)

    def _create_dynamic_knn_graph(self, sim_matrix, k_values, num_nodes):
        """创建动态K的KNN图"""
        edges = []
        for i in range(num_nodes):
            k_i = int(k_values[i])
            if k_i > 0 and k_i < num_nodes:
                neighbors = np.argpartition(-sim_matrix[i], kth=k_i)[:k_i]
                for j in neighbors:
                    if i != j:
                        edges.append((i, j))
        
        if edges:
            src, dst = zip(*edges)
            g = dgl.graph((src, dst), num_nodes=num_nodes).to(self._device)
        else:
            g = dgl.graph(([], []), num_nodes=num_nodes).to(self._device)
        
        return dgl.add_self_loop(g)

    def _generate_feat_graph(self):
        """使用训练好的模型生成最终的特征图"""
        print("Generating feature graphs with learned dynamic K values...")
        
        # 获取最终的嵌入和K值
        drug_features = torch.FloatTensor(self.drug_enhanced_features).to(self._device)
        disease_features = torch.FloatTensor(self.disease_enhanced_features).to(self._device)
        
        drug_stats = torch.FloatTensor(
            self._compute_graph_statistics(self.drug_enhanced_features)
        ).to(self._device)
        disease_stats = torch.FloatTensor(
            self._compute_graph_statistics(self.disease_enhanced_features)
        ).to(self._device)
        
        with torch.no_grad():
            # 创建临时图来获取嵌入
            temp_drug_graph = self._create_knn_graph(
                self.drug_sim_features, min(10, self.num_drug-1), self.num_drug
            )
            temp_disease_graph = self._create_knn_graph(
                self.disease_sim_features, min(10, self.num_disease-1), self.num_disease
            )
            
            # 获取嵌入
            drug_embeddings = self.drug_gat(temp_drug_graph, drug_features)
            disease_embeddings = self.disease_gat(temp_disease_graph, disease_features)
            
            # 预测K值
            drug_k_pred = self.drug_k_predictor(drug_embeddings, drug_stats)
            disease_k_pred = self.disease_k_predictor(disease_embeddings, disease_stats)
            
            # 转换为实际K值
            drug_k_values = drug_k_pred.cpu().numpy().flatten()
            disease_k_values = disease_k_pred.cpu().numpy().flatten()
            
            drug_dynamic_k = np.clip(
                np.round(drug_k_values * (self.k_max - self.k_min) + self.k_min),
                self.k_min,
                min(self.k_max, self.num_drug - 1)
            ).astype(int)
            
            disease_dynamic_k = np.clip(
                np.round(disease_k_values * (self.k_max - self.k_min) + self.k_min),
                self.k_min,
                min(self.k_max, self.num_disease - 1)
            ).astype(int)
        
        # 打印K值统计
        print(f"Drug K values - Min: {np.min(drug_dynamic_k)}, "
            f"Max: {np.max(drug_dynamic_k)}, "
            f"Mean: {np.mean(drug_dynamic_k):.2f}, "
            f"Std: {np.std(drug_dynamic_k):.2f}")
        print(f"Disease K values - Min: {np.min(disease_dynamic_k)}, "
            f"Max: {np.max(disease_dynamic_k)}, "
            f"Mean: {np.mean(disease_dynamic_k):.2f}, "
            f"Std: {np.std(disease_dynamic_k):.2f}")
        
        # 构建最终的加权图
        drug_edges, drug_weights = self._build_weighted_edges(
            self.drug_sim_features, drug_embeddings, drug_dynamic_k
        )
        disease_edges, disease_weights = self._build_weighted_edges(
            self.disease_sim_features, disease_embeddings, disease_dynamic_k
        )
        
        # 创建邻接矩阵
        drug_adj = self._create_adjacency_matrix(
            drug_edges, drug_weights, self.num_drug
        )
        disease_adj = self._create_adjacency_matrix(
            disease_edges, disease_weights, self.num_disease
        )
        
        # 处理对称性
        if self._symm:
            drug_adj = drug_adj + drug_adj.T.multiply(drug_adj.T > drug_adj) - \
                    drug_adj.multiply(drug_adj.T > drug_adj)
            disease_adj = disease_adj + disease_adj.T.multiply(disease_adj.T > disease_adj) - \
                        disease_adj.multiply(disease_adj.T > disease_adj)
        
        # 添加自环
        drug_adj = drug_adj + sp.eye(self.num_drug)
        disease_adj = disease_adj + sp.eye(self.num_disease)
        
        # 标准化
        drug_adj_norm = self._normalize_adj(drug_adj)
        disease_adj_norm = self._normalize_adj(disease_adj)
        
        # 转换为PyTorch稀疏张量
        drug_graph = self._sparse_mx_to_torch_sparse_tensor(drug_adj_norm)
        disease_graph = self._sparse_mx_to_torch_sparse_tensor(disease_adj_norm)
        
        print("Feature graphs generation completed.")
        return drug_graph, disease_graph

    def _build_weighted_edges(self, sim_matrix, embeddings, k_values):
        """构建加权边"""
        edges = []
        weights = []
        
        embeddings_np = embeddings.cpu().numpy()
        
        for i in range(len(k_values)):
            k_i = k_values[i]
            if k_i > 0:
                # 获取top-k邻居
                neighbors = np.argpartition(-sim_matrix[i], kth=k_i)[:k_i]
                
                for j in neighbors:
                    if i != j:
                        edges.append((i, j))
                        
                        # 计算权重：结合原始相似度和学习的嵌入相似度
                        orig_sim = sim_matrix[i, j]
                        embed_sim = np.dot(embeddings_np[i], embeddings_np[j]) / (
                            np.linalg.norm(embeddings_np[i]) * np.linalg.norm(embeddings_np[j]) + 1e-8
                        )
                        
                        # 加权组合
                        weight = 0.7 * orig_sim + 0.3 * max(0, embed_sim)
                        weights.append(weight)
        
        return edges, weights

    def _create_adjacency_matrix(self, edges, weights, num_nodes):
        """创建加权邻接矩阵"""
        if edges:
            src, dst = zip(*edges)
            adj = sp.coo_matrix(
                (weights, (src, dst)),
                shape=(num_nodes, num_nodes),
                dtype=np.float32
            )
        else:
            adj = sp.coo_matrix((num_nodes, num_nodes), dtype=np.float32)
        return adj

    def _normalize_adj(self, adj):
        """标准化邻接矩阵"""
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """将scipy稀疏矩阵转换为PyTorch稀疏张量"""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


    def _load_drug_data(self, file_path, data_name):
        association_matrix = None
        if data_name in ['Gdataset', 'Cdataset']:
            data = sio.loadmat(file_path)
            association_matrix = data['didr'].T
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']
        elif data_name in ['Ldataset']:
            association_matrix = np.loadtxt(os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
            self.disease_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
            self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            association_matrix = data.values
            self.disease_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values

        # self.disease_sim_features = th.FloatTensor(self.disease_sim_features)
        # self.drug_sim_features = th.FloatTensor(self.drug_sim_features)

        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]

        kfold = KFold(n_splits=10, shuffle=True, random_state=1024)
        pos_row, pos_col = np.nonzero(association_matrix)
        neg_row, neg_col = np.nonzero(1 - association_matrix)
        assert len(pos_row) + len(neg_row) == np.prod(association_matrix.shape)
        cv_num = 0
        cv_data = {}
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                kfold.split(neg_row)):
            train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
            train_pos_values = [1] * len(train_pos_edge[0])
            train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
            train_neg_values = [0] * len(train_neg_edge[0])

            test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
            test_pos_values = [1] * len(test_pos_edge[0])

            '''
            # test positive and test negative ration is 1:1
                test_neg_edge = np.stack([neg_row[test_neg_idx][0:len(test_pos_values)],
                                          neg_col[test_neg_idx][0:len(test_pos_values)]])

            '''

            test_neg_edge = np.stack([neg_row[test_neg_idx],
                                      neg_col[test_neg_idx]])
            test_neg_values = [0] * len(test_neg_edge[0])

            train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
            train_values = np.concatenate([train_pos_values, train_neg_values])
            test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
            test_values = np.concatenate([test_pos_values, test_neg_values])

            train_data = {
                'disease_id': train_edge[0],
                'drug_id': train_edge[1],
                'values': train_values
            }
            train_data_info = pd.DataFrame(train_data, index=None)

            test_data = {
                'disease_id': test_edge[0],
                'drug_id': test_edge[1],
                'values': test_values
            }
            test_data_info = pd.DataFrame(test_data, index=None)
            values = np.unique(train_values)
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data

    def _generate_feat(self):
        self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3)
        self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3)

        self.drug_feature = th.cat(
            [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
             th.zeros([self.num_drug, 1])], 1)

        self.disease_feature = th.cat(
            [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
             th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1)

    def _generate_topoy_graph(self):
        self.data_cv = {}
        for cv in range(0, 10):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values

            train_pairs, train_values = self._generate_pair_value(
                self.train_rel_info)
            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)

            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv
    


    
    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["disease_id"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["drug_id"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rel_values:
            ridx = np.where(
                rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('drug', str(rating), 'disease'): (rrow, rcol),
                ('disease', 'rev-%s' % str(rating), 'drug'): (rcol, rrow)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            drug_ci = []
            drug_cj = []
            disease_ci = []
            disease_cj = []
            for r in self.possible_rel_values:
                r = to_etype_name(r)
                drug_ci.append(graph['rev-%s' % r].in_degrees())
                disease_ci.append(graph[r].in_degrees())
                if self._symm:
                    drug_cj.append(graph[r].out_degrees())
                    disease_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    drug_cj.append(th.zeros((self.num_drug,)))
                    disease_cj.append(th.zeros((self.num_disease,)))

            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self.num_drug, )
                disease_cj = th.ones(self.num_disease, )
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                     vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug
    

class gaiImprovedGATLayer(nn.Module): 
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1): 
        super(gaiImprovedGATLayer, self).__init__() 
        self.num_heads = num_heads 
        self.out_features = out_features
        self.W = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False) 
            for _ in range(num_heads)
        ])
        self.a = nn.ModuleList([
            nn.Linear(2 * out_features, 1, bias=False)
            for _ in range(num_heads)
        ])
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_proj = nn.Linear(num_heads * out_features, out_features)
        
        # 稳定初始化
        for i in range(num_heads):
            nn.init.xavier_normal_(self.W[i].weight, gain=0.5)
            nn.init.xavier_normal_(self.a[i].weight, gain=0.5)
        nn.init.xavier_normal_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.out_proj.bias)

    def edge_attention(self, edges, head_idx):
        # 确保特征不包含NaN
        src_feat = torch.where(torch.isnan(edges.src['h']), 
                              torch.zeros_like(edges.src['h']), 
                              edges.src['h'])
        dst_feat = torch.where(torch.isnan(edges.dst['h']), 
                              torch.zeros_like(edges.dst['h']), 
                              edges.dst['h'])
        
        # 限制特征范围
        src_feat = torch.clamp(src_feat, -10, 10)
        dst_feat = torch.clamp(dst_feat, -10, 10)
        
        feat = torch.cat([src_feat, dst_feat], dim=1)
        attention = self.a[head_idx](feat)
        attention = self.leaky_relu(attention)
        
        # 数值稳定：避免exp爆炸
        return {'alpha': torch.exp(torch.clamp(attention, -5, 5))}
    
    def forward(self, g, features):
        # 检查输入特征
        features = torch.where(torch.isnan(features), 
                              torch.zeros_like(features), 
                              features)
        features = torch.clamp(features, -10, 10)
        
        h_heads = []
        for i in range(self.num_heads):
            h = self.W[i](features)
            
            with g.local_scope():
                g.ndata['h'] = h
                g.apply_edges(lambda edges: self.edge_attention(edges, i))
                g.edata['alpha'] = self.dropout(g.edata['alpha'])
                g.update_all(
                    fn.u_mul_e('h', 'alpha', 'm'),
                    fn.sum('m', 'h_agg')
                )
                h_head = g.ndata['h_agg']
                h_heads.append(h_head)
        
        h_multi = torch.cat(h_heads, dim=1)
        out = self.out_proj(h_multi)
        return F.elu(out)



class gaiAdaptiveKPredictor(nn.Module): 
    def __init__(self, in_features, hidden_dim=128, dataset_factor=1.0): 
        super(gaiAdaptiveKPredictor, self).__init__() 
        self.hidden_dim = hidden_dim
        self.in_features = in_features
        self.initialized = False
        # 新增：数据集特性因子，用于调整预测范围
        self.dataset_factor = dataset_factor
        
    def _initialize(self, input_dim):
        """更复杂的网络架构"""
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2) 
        self.bn2 = nn.BatchNorm1d(self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim // 4)
        self.fc4 = nn.Linear(self.hidden_dim // 4, 1)
        self.dropout = nn.Dropout(0.3)
        
        # 不同的初始化以产生多样化输出 - 修改初始化参数
        nn.init.xavier_uniform_(self.fc1.weight, gain=self.dataset_factor)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.8 * self.dataset_factor)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.8 * self.dataset_factor)
        nn.init.normal_(self.fc4.weight, mean=0.0, std=0.01 * self.dataset_factor)
        # 动态调整偏置初始化，使得不同数据集有不同的初始K分布
        nn.init.constant_(self.fc4.bias, 0.5 * self.dataset_factor)
        
        self.initialized = True

    def forward(self, node_features, graph_stats):
        # 处理输入
        node_features = torch.where(torch.isnan(node_features), 
                                  torch.zeros_like(node_features), 
                                  node_features)
        graph_stats = torch.where(torch.isnan(graph_stats), 
                                torch.zeros_like(graph_stats), 
                                graph_stats)
        
        # 添加噪声鼓励多样性 - 增加噪声量以产生更多样化的输出
        if self.training:
            noise = torch.randn_like(node_features) * 0.02 * self.dataset_factor
            node_features = node_features + noise
        
        x = torch.cat([node_features, graph_stats], dim=1)
        
        if not self.initialized:
            input_dim = x.size(1)
            self._initialize(input_dim)
            self.to(x.device)
        
        x = torch.clamp(x, -10, 10)
        
        batch_size = x.size(0)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        # 使用自定义激活函数产生更多样的输出
        x = 0.5 + 0.4 * self.dataset_factor * torch.tanh(self.fc4(x))
        
        return x

class gaiNewDrugDataLoader(object):
    def __init__(self,
             name,
             device,
             symm=True,
             learning_rate=0.001,
             k_min=1,
             k_max=20,
             attention_epochs=100,
            #  use_dynamic_power=True
            ):
        
        self._name = name
        self._device = device
        self._symm = symm
        self.k_min = k_min
        self.learning_rate=learning_rate
        
        print("Starting processing {} ...".format(self._name))
        self._dir = os.path.join(_paths[self._name])

        self.cv_data_dict = self._load_drug_data(self._dir, self._name)
        
        # 特征预处理
        self._preprocess_features()
        
        # 计算数据集特性因子 - 新增
        self.dataset_density = self._compute_dataset_density()
        print(f"数据集 {self._name} 的密度因子: {self.dataset_density:.4f}")
        
        # 基于数据集特性调整学习率 - 新增
        adjusted_lr = self.learning_rate * (0.5 + 0.5 * self.dataset_density)
        print(f"基于数据集特性调整学习率: {adjusted_lr:.6f}")
        
        # 根据数据集特性调整K最大值 - 新增
        self.k_max_adjusted = max(k_min + 2, min(int(k_max * self.dataset_density), self.num_drug - 1))
        print(f"调整后的K最大值: {self.k_max_adjusted}")
        
        # 初始化改进的GAT层
        self.drug_gat = gaiImprovedGATLayer(
            self.drug_feature_dim, 
            self.drug_feature_dim, 
            num_heads=4
        ).to(device)
        
        self.disease_gat = gaiImprovedGATLayer(
            self.disease_feature_dim,
            self.disease_feature_dim,
            num_heads=4
        ).to(device)
        
        # 初始化K预测器 - 传入数据集特性因子
        self.drug_k_predictor = gaiAdaptiveKPredictor(
            self.drug_feature_dim,
            dataset_factor=self.dataset_density
        ).to(device)
        
        self.disease_k_predictor = gaiAdaptiveKPredictor(
            self.disease_feature_dim,
            dataset_factor=self.dataset_density
        ).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.drug_gat.parameters(), 'lr': adjusted_lr},
            {'params': self.disease_gat.parameters(), 'lr': adjusted_lr},
            {'params': self.drug_k_predictor.parameters(), 'lr': adjusted_lr * 0.1},
            {'params': self.disease_k_predictor.parameters(), 'lr': adjusted_lr * 0.1}
        ], weight_decay=1e-5)
        
        # 训练模型
        self._train_models(epochs=attention_epochs)
        
        # 生成图
        self._generate_topoy_graph()
        self.drug_graph, self.disease_graph = self._generate_feat_graph()
        self._generate_feat()
        self.possible_rel_values = self.values

    def _compute_dataset_density(self):
        """计算数据集的密度特性"""
        # 计算相似度矩阵的特性
        drug_density = np.mean(self.drug_sim_features > 0.5)
        disease_density = np.mean(self.disease_sim_features > 0.5)
        
        # 计算数据集规模因子
        scale_factor = min(1.0, np.sqrt((self.num_drug + self.num_disease) / 1000))
        
        # 结合密度和规模
        density_factor = (drug_density + disease_density) / 2
        
        # 返回调整后的数据集特性因子(0.5-1.5范围)
        return 0.5 + density_factor + 0.2 * scale_factor

    def _preprocess_features(self):
        """特征预处理和增强"""
        # 标准化相似度矩阵
        scaler = StandardScaler()
        
        # 药物特征增强
        drug_sim_flat = self.drug_sim_features.reshape(-1, 1)
        drug_sim_normalized = scaler.fit_transform(drug_sim_flat).reshape(self.drug_sim_features.shape)
        
        # 计算额外的统计特征
        drug_degree = np.sum(self.drug_sim_features > 0.5, axis=1, keepdims=True)
        drug_avg_sim = np.mean(self.drug_sim_features, axis=1, keepdims=True)
        drug_std_sim = np.std(self.drug_sim_features, axis=1, keepdims=True)
        
        # 拼接特征
        self.drug_enhanced_features = np.concatenate([
            drug_sim_normalized,
            drug_degree / self.num_drug,  # 归一化度
            drug_avg_sim,
            drug_std_sim
        ], axis=1)
        
        # 疾病特征增强
        disease_sim_flat = self.disease_sim_features.reshape(-1, 1)
        disease_sim_normalized = scaler.fit_transform(disease_sim_flat).reshape(self.disease_sim_features.shape)
        
        disease_degree = np.sum(self.disease_sim_features > 0.5, axis=1, keepdims=True)
        disease_avg_sim = np.mean(self.disease_sim_features, axis=1, keepdims=True)
        disease_std_sim = np.std(self.disease_sim_features, axis=1, keepdims=True)
        
        self.disease_enhanced_features = np.concatenate([
            disease_sim_normalized,
            disease_degree / self.num_disease,
            disease_avg_sim,
            disease_std_sim
        ], axis=1)
        
        self.drug_feature_dim = self.drug_enhanced_features.shape[1]
        self.disease_feature_dim = self.disease_enhanced_features.shape[1]

    def _compute_graph_statistics(self, features):
        """计算图的全局统计信息"""
        stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'max': np.max(features, axis=0),
            'min': np.min(features, axis=0)
        }
        # 广播到每个节点
        graph_stats = np.concatenate([
            np.tile(stats['mean'], (features.shape[0], 1)),
            np.tile(stats['std'], (features.shape[0], 1)),
        ], axis=1)
        return graph_stats

    def _compute_k_targets(self, sim_matrix, k_min, k_max):
        """改进的目标K值计算，基于数据集特性自适应调整阈值"""
        n = sim_matrix.shape[0]
        k_targets = []
        
        # 计算数据集特定的阈值 - 新增
        avg_sim = np.mean(sim_matrix)
        sim_std = np.std(sim_matrix)
        
        # 动态阈值计算
        high_sim_threshold = max(0.5, avg_sim + 0.5 * sim_std)
        med_sim_threshold = max(0.3, avg_sim)
        sig_drop_threshold = -1 * min(0.1, 0.5 * sim_std)  # 数据集相似度波动小时使用更小的阈值
        
        print(f"数据集自适应阈值 - 高相似度: {high_sim_threshold:.3f}, 中相似度: {med_sim_threshold:.3f}, 显著下降: {sig_drop_threshold:.3f}")
        
        # 为每个节点找到合适的K
        for i in range(n):
            # 分析节点i的连接模式
            sims = sim_matrix[i]
            sorted_sims = np.sort(sims)[::-1]
            
            # 使用数据集自适应的相似度下降阈值
            sim_diffs = np.diff(sorted_sims)
            significant_drops = np.where(sim_diffs < sig_drop_threshold)[0]
            
            # 使用自适应的高/中相似度阈值
            high_sim_ratio = np.sum(sims > high_sim_threshold) / n
            med_sim_ratio = np.sum((sims > med_sim_threshold) & (sims <= high_sim_threshold)) / n
            
            # 节点度中心性 - 新增
            degree_centrality = np.sum(sims > med_sim_threshold) / n
            
            if len(significant_drops) > 0:
                # 如果存在明显的相似度下降点，使用它作为K值的基础
                elbow_point = significant_drops[0] + 1
                # 结合节点的度中心性进行调整
                target_k = (0.3 + 0.6 * degree_centrality) * (elbow_point - k_min) / (k_max - k_min)
            else:
                # 基于相似度分布特性和节点中心性
                if high_sim_ratio > 0.1:
                    # 高相似度节点获得较高的K值
                    target_k = 0.6 + high_sim_ratio * 0.4 + 0.1 * degree_centrality
                elif med_sim_ratio > 0.2:
                    # 中等相似度节点获得中等K值
                    target_k = 0.3 + med_sim_ratio * 0.4 + 0.1 * degree_centrality
                else:
                    # 低相似度节点获得较低但非零的K值
                    target_k = 0.1 + 0.2 * degree_centrality
            
            # 应用数据集密度因子调整目标K值
            target_k = target_k * self.dataset_density
            
            # 动态上下限，确保有足够的多样性
            min_k_target = max(0.05, 0.1 * self.dataset_density)
            max_k_target = min(0.95, 0.8 + 0.2 * self.dataset_density)
            
            k_targets.append(np.clip(target_k, min_k_target, max_k_target))
        
        # 输出K值分布统计
        targets_array = np.array(k_targets)
        print(f"目标K值统计 - 最小值: {np.min(targets_array):.3f}, 最大值: {np.max(targets_array):.3f}, "
              f"平均值: {np.mean(targets_array):.3f}, 标准差: {np.std(targets_array):.3f}")
        
        return np.array(k_targets)

    def _train_models(self, epochs=100):
        """改进的训练过程，让K值自然反映节点特性"""
        print("训练注意力机制和K预测器...")
        
        # 准备训练数据
        drug_features = torch.FloatTensor(self.drug_enhanced_features).to(self._device)
        disease_features = torch.FloatTensor(self.disease_enhanced_features).to(self._device)
        
        # 计算图统计信息
        drug_stats = torch.FloatTensor(
            self._compute_graph_statistics(self.drug_enhanced_features)
        ).to(self._device)
        disease_stats = torch.FloatTensor(
            self._compute_graph_statistics(self.disease_enhanced_features)
        ).to(self._device)
        
        # 检查并替换NaN值
        drug_features = torch.where(torch.isnan(drug_features), 
                                torch.zeros_like(drug_features), 
                                drug_features)
        disease_features = torch.where(torch.isnan(disease_features), 
                                    torch.zeros_like(disease_features), 
                                    disease_features)
        drug_stats = torch.where(torch.isnan(drug_stats), 
                            torch.zeros_like(drug_stats), 
                            drug_stats)
        disease_stats = torch.where(torch.isnan(disease_stats), 
                                torch.zeros_like(disease_stats), 
                                disease_stats)
        
        # 根据数据集规模自适应初始K值
        init_k = max(1, min(5, int(self.num_drug * 0.05)))
        print(f"初始化KNN图使用K={init_k}")
        
        drug_graph = self._create_knn_graph(self.drug_sim_features, init_k, self.num_drug)
        disease_graph = self._create_knn_graph(self.disease_sim_features, init_k, self.num_disease)
        
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        epsilon = 1e-8  # 用于数值稳定性
        
        # 使用学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 计算目标K值 - 基于数据集特性
        drug_k_target = self._compute_k_targets(
            self.drug_sim_features, self.k_min, self.k_max_adjusted
        )
        disease_k_target = self._compute_k_targets(
            self.disease_sim_features, self.k_min, self.k_max_adjusted
        )
        
        drug_k_target = torch.FloatTensor(drug_k_target).unsqueeze(1).to(self._device)
        disease_k_target = torch.FloatTensor(disease_k_target).unsqueeze(1).to(self._device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # 前向传播 - GAT
            drug_embeddings = self.drug_gat(drug_graph, drug_features)
            disease_embeddings = self.disease_gat(disease_graph, disease_features)
            
            # 检查嵌入是否包含NaN
            if torch.isnan(drug_embeddings).any() or torch.isnan(disease_embeddings).any():
                print("警告: 检测到NaN嵌入")
                drug_embeddings = torch.where(torch.isnan(drug_embeddings), 
                                            torch.zeros_like(drug_embeddings), 
                                            drug_embeddings)
                disease_embeddings = torch.where(torch.isnan(disease_embeddings), 
                                                torch.zeros_like(disease_embeddings), 
                                                disease_embeddings)
            
            # 计算重建损失
            drug_sim_pred = torch.mm(drug_embeddings, drug_embeddings.t())
            disease_sim_pred = torch.mm(disease_embeddings, disease_embeddings.t())
            
            # 使用加权损失，更重视高相似度的边
            drug_sim_true = torch.FloatTensor(self.drug_sim_features).to(self._device)
            disease_sim_true = torch.FloatTensor(self.disease_sim_features).to(self._device)
            
            # 加权MSE损失 - 调整权重计算
            drug_weights = 1 + drug_sim_true * 3 * self.dataset_density  # 高相似度边权重更大，且受数据集密度影响
            disease_weights = 1 + disease_sim_true * 3 * self.dataset_density
            
            drug_recon_loss = torch.mean(drug_weights * (drug_sim_pred - drug_sim_true) ** 2 + epsilon)
            disease_recon_loss = torch.mean(disease_weights * (disease_sim_pred - disease_sim_true) ** 2 + epsilon)
            
            # K预测损失
            drug_k_pred = self.drug_k_predictor(drug_embeddings, drug_stats)
            disease_k_pred = self.disease_k_predictor(disease_embeddings, disease_stats)
            
            # 检查预测值是否包含NaN
            if torch.isnan(drug_k_pred).any() or torch.isnan(disease_k_pred).any():
                print("警告: 检测到NaN的K预测值")
                drug_k_pred = torch.where(torch.isnan(drug_k_pred), 
                                        torch.ones_like(drug_k_pred) * 0.5, 
                                        drug_k_pred)
                disease_k_pred = torch.where(torch.isnan(disease_k_pred), 
                                            torch.ones_like(disease_k_pred) * 0.5, 
                                            disease_k_pred)
            
            # 基本K预测损失
            k_mse_loss = F.mse_loss(drug_k_pred, drug_k_target) + F.mse_loss(disease_k_pred, disease_k_target)
            
            # 多样性促进损失 - 新增
            drug_k_diversity = -0.2 * torch.std(drug_k_pred)
            disease_k_diversity = -0.2 * torch.std(disease_k_pred)
            diversity_loss = drug_k_diversity + disease_k_diversity
            
            # 总K预测损失
            k_loss = k_mse_loss + diversity_loss
            
            # 输出当前预测的K值分布
            if epoch % 10 == 0:
                drug_k_mean = torch.mean(drug_k_pred).item()
                drug_k_std = torch.std(drug_k_pred).item()
                disease_k_mean = torch.mean(disease_k_pred).item()
                disease_k_std = torch.std(disease_k_pred).item()
                print(f"  K预测值 - 药物: 均值={drug_k_mean:.3f}, 标准差={drug_k_std:.3f} | "
                    f"疾病: 均值={disease_k_mean:.3f}, 标准差={disease_k_std:.3f}")
                print(f"  多样性损失: {diversity_loss.item():.4f}")
            
            # 总损失
            total_loss = drug_recon_loss + disease_recon_loss + 0.2 * self.dataset_density * k_loss + epsilon
            
            # 检查损失是否为NaN
            if torch.isnan(total_loss).any():
                print(f"警告: 第{epoch+1}轮检测到NaN损失")
                # 使用上一次的有效损失或设置默认损失
                total_loss = torch.tensor(1.0, requires_grad=True, device=self._device)
            
            # 添加正则化
            l2_reg = 0.0001 * sum(p.pow(2).sum() for p in self.drug_gat.parameters())
            total_loss += l2_reg
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.drug_gat.parameters()) + 
                list(self.disease_gat.parameters()) +
                list(self.drug_k_predictor.parameters()) +
                list(self.disease_k_predictor.parameters()), 
                max_norm=0.5
            )

            self.optimizer.step()
            
            # 更新学习率调度器
            scheduler.step(total_loss)
            
            # 更新图结构的频率基于数据集规模动态调整
            update_freq = max(5, min(15, int(20 / self.dataset_density)))
            
            # 每update_freq轮更新一次图结构
            if (epoch + 1) % update_freq == 0:
                with torch.no_grad():
                    # 使用当前预测的K值更新图
                    drug_k_values = drug_k_pred.cpu().numpy().flatten()
                    disease_k_values = disease_k_pred.cpu().numpy().flatten()
                    
                    # 使用非线性映射转换K值 - 新增
                    # 幂次方数基于数据集密度，较低密度使用更高幂次
                    power_factor = 1.0 / max(0.5, self.dataset_density)
                    
                    # 将0-1范围的预测值转换为实际K值，使用非线性映射
                    drug_k_int = np.clip(
                        np.round(self.k_min + (self.k_max_adjusted - self.k_min) * np.power(drug_k_values, power_factor)),
                        self.k_min, 
                        min(self.k_max_adjusted, self.num_drug - 1)
                    ).astype(int)
                    
                    disease_k_int = np.clip(
                        np.round(self.k_min + (self.k_max_adjusted - self.k_min) * np.power(disease_k_values, power_factor)),
                        self.k_min,
                        min(self.k_max_adjusted, self.num_disease - 1)
                    ).astype(int)
                    
                    # 检查K值是否有效
                    if np.any(drug_k_int <= 0) or np.any(disease_k_int <= 0):
                        print("警告: 检测到无效K值")
                        drug_k_int = np.maximum(drug_k_int, 1)
                        disease_k_int = np.maximum(disease_k_int, 1)
                    
                    # 输出K值统计
                    print(f"  当前K值 - 药物: 最小={np.min(drug_k_int)}, 最大={np.max(drug_k_int)}, "
                          f"均值={np.mean(drug_k_int):.2f}, 标准差={np.std(drug_k_int):.2f}")
                    print(f"  当前K值 - 疾病: 最小={np.min(disease_k_int)}, 最大={np.max(disease_k_int)}, "
                          f"均值={np.mean(disease_k_int):.2f}, 标准差={np.std(disease_k_int):.2f}")
                    
                    # 使用动态K值创建新图
                    drug_graph = self._create_dynamic_knn_graph(
                        self.drug_sim_features, drug_k_int, self.num_drug
                    )
                    disease_graph = self._create_dynamic_knn_graph(
                        self.disease_sim_features, disease_k_int, self.num_disease
                    )
            
            # 早停
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"在第{epoch + 1}轮提前停止训练")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"轮次 {epoch + 1}/{epochs}, 总损失: {total_loss.item():.4f}, "
                    f"重建损失: {(drug_recon_loss + disease_recon_loss).item():.4f}, "
                    f"K损失: {k_loss.item():.4f}")
        
        # 设置为评估模式
        self.drug_gat.eval()
        self.disease_gat.eval()
        self.drug_k_predictor.eval()
        self.disease_k_predictor.eval()
        
        print("训练完成。")



    def _create_knn_graph(self, sim_matrix, k, num_nodes):
        """创建固定K的KNN图"""
        edges = []
        for i in range(num_nodes):
            # 获取top-k邻居
            if k < num_nodes:
                neighbors = np.argpartition(-sim_matrix[i], kth=k)[:k]
            else:
                neighbors = np.arange(num_nodes)
            
            for j in neighbors:
                if i != j:
                    edges.append((i, j))
        
        if edges:
            src, dst = zip(*edges)
            g = dgl.graph((src, dst), num_nodes=num_nodes).to(self._device)
        else:
            g = dgl.graph(([], []), num_nodes=num_nodes).to(self._device)
        
        return dgl.add_self_loop(g)

    def _create_dynamic_knn_graph(self, sim_matrix, k_values, num_nodes):
        """创建动态K的KNN图"""
        edges = []
        for i in range(num_nodes):
            k_i = int(k_values[i])
            if k_i > 0 and k_i < num_nodes:
                neighbors = np.argpartition(-sim_matrix[i], kth=k_i)[:k_i]
                for j in neighbors:
                    if i != j:
                        edges.append((i, j))
        
        if edges:
            src, dst = zip(*edges)
            g = dgl.graph((src, dst), num_nodes=num_nodes).to(self._device)
        else:
            g = dgl.graph(([], []), num_nodes=num_nodes).to(self._device)
        
        return dgl.add_self_loop(g)

    def _generate_feat_graph(self):
        """使用训练好的模型生成最终的特征图"""
        print("使用学习到的动态K值生成特征图...")
        
        # 获取最终的嵌入和K值
        drug_features = torch.FloatTensor(self.drug_enhanced_features).to(self._device)
        disease_features = torch.FloatTensor(self.disease_enhanced_features).to(self._device)
        
        drug_stats = torch.FloatTensor(
            self._compute_graph_statistics(self.drug_enhanced_features)
        ).to(self._device)
        disease_stats = torch.FloatTensor(
            self._compute_graph_statistics(self.disease_enhanced_features)
        ).to(self._device)
        
        with torch.no_grad():
            # 创建临时图来获取嵌入
            temp_drug_graph = self._create_knn_graph(
                self.drug_sim_features, min(10, self.num_drug-1), self.num_drug
            )
            temp_disease_graph = self._create_knn_graph(
                self.disease_sim_features, min(10, self.num_disease-1), self.num_disease
            )
            
            # 获取嵌入
            drug_embeddings = self.drug_gat(temp_drug_graph, drug_features)
            disease_embeddings = self.disease_gat(temp_disease_graph, disease_features)
            
            # 预测K值
            drug_k_pred = self.drug_k_predictor(drug_embeddings, drug_stats)
            disease_k_pred = self.disease_k_predictor(disease_embeddings, disease_stats)
            
            # 转换为实际K值 - 使用非线性映射
            drug_k_values = drug_k_pred.cpu().numpy().flatten()
            disease_k_values = disease_k_pred.cpu().numpy().flatten()
            
            # 使用与训练相同的非线性映射
            power_factor = 1.0 / max(0.5, self.dataset_density)
            
            drug_dynamic_k = np.clip(
                np.round(self.k_min + (self.k_max_adjusted - self.k_min) * np.power(drug_k_values, power_factor)),
                self.k_min,
                min(self.k_max_adjusted, self.num_drug - 1)
            ).astype(int)
            
            disease_dynamic_k = np.clip(
                np.round(self.k_min + (self.k_max_adjusted - self.k_min) * np.power(disease_k_values, power_factor)),
                self.k_min,
                min(self.k_max_adjusted, self.num_disease - 1)
            ).astype(int)
        
        # 打印K值统计
        print(f"药物K值 - 最小: {np.min(drug_dynamic_k)}, "
            f"最大: {np.max(drug_dynamic_k)}, "
            f"均值: {np.mean(drug_dynamic_k):.2f}, "
            f"标准差: {np.std(drug_dynamic_k):.2f}")
        print(f"疾病K值 - 最小: {np.min(disease_dynamic_k)}, "
            f"最大: {np.max(disease_dynamic_k)}, "
            f"均值: {np.mean(disease_dynamic_k):.2f}, "
            f"标准差: {np.std(disease_dynamic_k):.2f}")
        
        # 构建最终的加权图
        drug_edges, drug_weights = self._build_weighted_edges(
            self.drug_sim_features, drug_embeddings, drug_dynamic_k
        )
        disease_edges, disease_weights = self._build_weighted_edges(
            self.disease_sim_features, disease_embeddings, disease_dynamic_k
        )
        
        # 创建邻接矩阵
        drug_adj = self._create_adjacency_matrix(
            drug_edges, drug_weights, self.num_drug
        )
        disease_adj = self._create_adjacency_matrix(
            disease_edges, disease_weights, self.num_disease
        )
        
        # 处理对称性
        if self._symm:
            drug_adj = drug_adj + drug_adj.T.multiply(drug_adj.T > drug_adj) - \
                    drug_adj.multiply(drug_adj.T > drug_adj)
            disease_adj = disease_adj + disease_adj.T.multiply(disease_adj.T > disease_adj) - \
                        disease_adj.multiply(disease_adj.T > disease_adj)
        
        # 添加自环
        drug_adj = drug_adj + sp.eye(self.num_drug)
        disease_adj = disease_adj + sp.eye(self.num_disease)
        
        # 标准化
        drug_adj_norm = self._normalize_adj(drug_adj)
        disease_adj_norm = self._normalize_adj(disease_adj)
        
        # 转换为PyTorch稀疏张量
        drug_graph = self._sparse_mx_to_torch_sparse_tensor(drug_adj_norm)
        disease_graph = self._sparse_mx_to_torch_sparse_tensor(disease_adj_norm)
        
        print("特征图生成完成。")
        return drug_graph, disease_graph

    def _build_weighted_edges(self, sim_matrix, embeddings, k_values):
        """构建加权边"""
        edges = []
        weights = []
        
        # embeddings_np = embeddings.cpu().numpy()
        
        for i in range(len(k_values)):
            k_i = k_values[i]
            if k_i > 0:
                # 获取top-k邻居
                neighbors = np.argpartition(-sim_matrix[i], kth=k_i)[:k_i]
                
                for j in neighbors:
                    if i != j:
                        edges.append((i, j))
                        
                        # 计算权重：结合原始相似度和学习的嵌入相似度
                        orig_sim = sim_matrix[i, j]
                        # embed_sim = np.dot(embeddings_np[i], embeddings_np[j]) / (
                        #     np.linalg.norm(embeddings_np[i]) * np.linalg.norm(embeddings_np[j]) + 1e-8
                        # )
                        
                        # 加权组合
                        weight =  orig_sim
                        weights.append(weight)
        
        
        return edges, weights

    def _create_adjacency_matrix(self, edges, weights, num_nodes):
        """创建加权邻接矩阵"""
        if edges:
            src, dst = zip(*edges)
            adj = sp.coo_matrix(
                (weights, (src, dst)),
                shape=(num_nodes, num_nodes),
                dtype=np.float32
            )
        else:
            adj = sp.coo_matrix((num_nodes, num_nodes), dtype=np.float32)
        return adj

    def _normalize_adj(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    @staticmethod
    def generate_twodrug_twodisease(input_mat_path: str, output_mat_path: str,drug_n,disease_n,fordrug_m,fordisease_m):
        """
        从input_mat_path加载.mat文件，计算twodrug和twodisease矩阵，
        保存结果到output_mat_path，并打印相关信息。
        """
        try:
            # 1. 加载数据
            data = sio.loadmat(input_mat_path)
            # 获取药物-疾病关联矩阵 didr
            # didr = data['didr']  # 形状 (313, 593)
            
            # # 计算疾病关联矩阵（313x313）
            # fordisease = np.dot(didr, didr.T)

            # # 计算药物关联矩阵（593x593）
            # fordrug = np.dot(didr.T, didr)    

            # 添加到数据字典
            fordisease = data['fordisease']
            fordrug = data['fordrug']
            # fordrug = data['fordrug']
            # fordisease = data['fordisease']
            drug = data['drug']
            disease = data['disease']

            # # # 2. 打印对称性和样本数据
            # # print("drug是否对称:", np.allclose(drug, drug.T, atol=1e-8))
            # # print(f"disease样本数据: {disease[:8] if disease.size >= 5 else disease}")
            # # print(f"fordisease样本数据: {fordisease[:8] if fordisease.size >= 5 else fordisease}")

            # # 3. 计算20次方根
            # drug_transformed = np.power(drug, 1/drug_n)
            # disease_transformed = np.power(disease, 1/disease_n)
            # # print(f"disease_transformed样本数据: {disease_transformed[:8] if disease_transformed.size >= 5 else disease_transformed}")

            # # 4. fordrug和fordisease归一化后开10次方根
            # max_val_drug = np.max(fordrug)
            # fordrug_normalized = np.power(fordrug / max_val_drug, 1/fordrug_m)
            # max_val_disease = np.max(fordisease)
            # fordisease_normalized = np.power(fordisease / max_val_disease, 1/fordisease_m)
            # # print(f"fordisease_normalized样本数据: {fordisease_normalized[:8] if fordisease_normalized.size >= 5 else fordisease_normalized}")

            # # # 5. 对应元素相乘生成twodrug和twodisease
            # # twodrug = 0.9*fordrug_normalized + 0.1*drug_transformed
            # # twodisease = 0.9*fordisease_normalized + 0.1*disease_transformed
            # twodrug = fordrug_normalized *drug_transformed
            # twodisease = fordisease_normalized *disease_transformed
            twodrug = fordrug
            twodisease = fordisease

            # 6. 保存到data字典
            data['twodrug'] = twodrug
            data['twodisease'] = twodisease

            # 7. 打印twodrug和twodisease信息
            for key in ['twodrug', 'twodisease']:
                var = data[key]
                print(f"Variable '{key}': type={type(var)}")
                if hasattr(var, 'shape'):
                    print(f"  shape: {var.shape}")
                print(f"  sample data: {var[:5] if var.size >= 5 else var}")

            # 8. 保存回.mat文件
            sio.savemat(output_mat_path, data)
            print(f"twodrug和twodisease矩阵已生成并保存到 {output_mat_path}")
        except Exception as e:
            print(f"生成twodrug_twodisease时出错: {e}")
            raise

    def _load_drug_data(self, file_path, data_name):
            association_matrix = None
            if data_name in ['Gdataset', 'Cdataset','twoGdataset','twoCdataset']:
                if data_name == 'twoGdataset' or data_name == 'twoCdataset':
                    if not os.path.exists(file_path):
                        path='/home/liujin/data/DRGCL-main/DRGCL-main/raw_data/drug_data/Cdataset/newCdataset.mat'
                        self.generate_twodrug_twodisease(path, file_path,drug_n=1,disease_n=1,fordrug_m=1,fordisease_m=1)
                    
                    # 如果文件已存在，仍然需要加载数据
                    data = sio.loadmat(file_path)
                    association_matrix = data['didr'].T
                    self.disease_sim_features = data['twodisease']
                    self.drug_sim_features = data['twodrug']

                else:    
                    data = sio.loadmat(file_path)
                    association_matrix = data['didr'].T
                    self.disease_sim_features = data['disease']
                    self.drug_sim_features = data['drug']

            elif data_name in ['lrssl','twolrssl','newlrssl']:
                if data_name == 'newlrssl'or data_name == 'twolrssl':
                    if not os.path.exists(file_path):
                        path='/home/liujin/data/DRGCL-main/DRGCL-main/raw_data/drug_data/lrssl/newlrssl.mat'
                        self.generate_twodrug_twodisease(path, file_path,drug_n=1,disease_n=1,fordrug_m=1,fordisease_m=1)
                    
                    # 如果文件已存在，仍然需要加载数据
                    data = sio.loadmat(file_path)
                    association_matrix = data['didr']
                    self.disease_sim_features = data['twodisease']
                    self.drug_sim_features = data['twodrug']
                else:  
                    data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
                    association_matrix = data.values
                    self.disease_sim_features = pd.read_csv(
                        os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
                    self.drug_sim_features = pd.read_csv(
                        os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values
                    

            self._num_drug = association_matrix.shape[0]
            self._num_disease = association_matrix.shape[1]

            # KFold cross-validation (原有的代码)
            kfold = KFold(n_splits=10, shuffle=True, random_state=1024)
            pos_row, pos_col = np.nonzero(association_matrix)
            neg_row, neg_col = np.nonzero(1 - association_matrix)
            assert len(pos_row) + len(neg_row) == np.prod(association_matrix.shape)
            cv_num = 0
            cv_data = {}
            for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                    kfold.split(neg_row)):
                train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
                train_pos_values = [1] * len(train_pos_edge[0])
                train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
                train_neg_values = [0] * len(train_neg_edge[0])

                test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
                test_pos_values = [1] * len(test_pos_edge[0])

                test_neg_edge = np.stack([neg_row[test_neg_idx],
                                        neg_col[test_neg_idx]])
                test_neg_values = [0] * len(test_neg_edge[0])

                train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
                train_values = np.concatenate([train_pos_values, train_neg_values])
                test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
                test_values = np.concatenate([test_pos_values, test_neg_values])

                train_data = {
                    'disease_id': train_edge[0],
                    'drug_id': train_edge[1],
                    'values': train_values
                }
                train_data_info = pd.DataFrame(train_data, index=None)

                test_data = {
                    'disease_id': test_edge[0],
                    'drug_id': test_edge[1],
                    'values': test_values
                }
                test_data_info = pd.DataFrame(test_data, index=None)
                values = np.unique(train_values)
                cv_data[cv_num] = [train_data_info, test_data_info, values]
                cv_num += 1

            return cv_data
    
    # def _load_drug_data(self, file_path, data_name):
    #     association_matrix = None
    #     if data_name in ['Gdataset', 'Cdataset','twoGdataset','twoCdataset']:
    #         if data_name == 'twoGdataset' or data_name == 'twoCdataset':
    #             if not os.path.exists(file_path):
    #                 path='/home/liujin/data/DRGCL-main/DRGCL-main/raw_data/drug_data/Cdataset/newCdataset.mat'
    #                 self.generate_twodrug_twodisease(path, file_path,drug_n=1,disease_n=1,fordrug_m=1,fordisease_m=1)
                
    #             # 如果文件已存在，仍然需要加载数据
    #             data = sio.loadmat(file_path)
    #             association_matrix = data['didr'].T
    #             self.disease_sim_features = data['twodisease']
    #             self.drug_sim_features = data['twodrug']

    #         else:    
    #             data = sio.loadmat(file_path)
    #             association_matrix = data['didr'].T
    #             self.disease_sim_features = data['disease']
    #             self.drug_sim_features = data['drug']

    #     elif data_name in ['lrssl','twolrssl','newlrssl']:
    #         if data_name == 'newlrssl'or data_name == 'twolrssl':
    #             if not os.path.exists(file_path):
    #                 path='/home/liujin/data/DRGCL-main/DRGCL-main/raw_data/drug_data/lrssl/newlrssl.mat'
    #                 self.generate_twodrug_twodisease(path, file_path,drug_n=1,disease_n=1,fordrug_m=1,fordisease_m=1)
                
    #             # 如果文件已存在，仍然需要加载数据
    #             data = sio.loadmat(file_path)
    #             association_matrix = data['didr']
    #             self.disease_sim_features = data['twodisease']
    #             self.drug_sim_features = data['twodrug']
    #         else:  
    #             data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
    #             association_matrix = data.values
    #             self.disease_sim_features = pd.read_csv(
    #                 os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
    #             self.drug_sim_features = pd.read_csv(
    #                 os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values
                

    #     self._num_drug = association_matrix.shape[0]
    #     self._num_disease = association_matrix.shape[1]

    #     # 获取所有正样本和负样本的位置
    #     pos_row, pos_col = np.nonzero(association_matrix)
    #     neg_row, neg_col = np.nonzero(1 - association_matrix)
        
    #     # 对正样本进行90%采样
    #     np.random.seed(1024)  # 设置随机种子以确保结果可重复
    #     pos_indices = np.arange(len(pos_row))
    #     selected_pos_indices = np.random.choice(pos_indices, 
    #                                         size=int(len(pos_indices) * 0.8), 
    #                                         replace=False)
        
    #     # 只使用采样后的正样本
    #     selected_pos_row = pos_row[selected_pos_indices]
    #     selected_pos_col = pos_col[selected_pos_indices]
        
    #     # 确保正样本+负样本的总数不超过矩阵大小
    #     assert len(selected_pos_row) + len(neg_row) <= np.prod(association_matrix.shape)
        
    #     # KFold cross-validation
    #     kfold = KFold(n_splits=10, shuffle=True, random_state=1024)
    #     cv_num = 0
    #     cv_data = {}
        
    #     for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(selected_pos_row),
    #                                                                             kfold.split(neg_row)):
    #         train_pos_edge = np.stack([selected_pos_row[train_pos_idx], selected_pos_col[train_pos_idx]])
    #         train_pos_values = [1] * len(train_pos_edge[0])
    #         train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
    #         train_neg_values = [0] * len(train_neg_edge[0])

    #         test_pos_edge = np.stack([selected_pos_row[test_pos_idx], selected_pos_col[test_pos_idx]])
    #         test_pos_values = [1] * len(test_pos_edge[0])

    #         test_neg_edge = np.stack([neg_row[test_neg_idx],
    #                                 neg_col[test_neg_idx]])
    #         test_neg_values = [0] * len(test_neg_edge[0])

    #         train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
    #         train_values = np.concatenate([train_pos_values, train_neg_values])
    #         test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
    #         test_values = np.concatenate([test_pos_values, test_neg_values])

    #         train_data = {
    #             'disease_id': train_edge[0],
    #             'drug_id': train_edge[1],
    #             'values': train_values
    #         }
    #         train_data_info = pd.DataFrame(train_data, index=None)

    #         test_data = {
    #             'disease_id': test_edge[0],
    #             'drug_id': test_edge[1],
    #             'values': test_values
    #         }
    #         test_data_info = pd.DataFrame(test_data, index=None)
    #         values = np.unique(train_values)
    #         cv_data[cv_num] = [train_data_info, test_data_info, values]
    #         cv_num += 1

    #     return cv_data


    def _generate_feat(self):
        self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3)
        self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3)

        self.drug_feature = th.cat(
            [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
             th.zeros([self.num_drug, 1])], 1)

        self.disease_feature = th.cat(
            [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
             th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1)

    def _generate_topoy_graph(self):
        self.data_cv = {}
        for cv in range(0, 10):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values

            train_pairs, train_values = self._generate_pair_value(
                self.train_rel_info)
            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)

            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv

    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["disease_id"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["drug_id"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rel_values:
            ridx = np.where(
                rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('drug', str(rating), 'disease'): (rrow, rcol),
                ('disease', 'rev-%s' % str(rating), 'drug'): (rcol, rrow)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            drug_ci = []
            drug_cj = []
            disease_ci = []
            disease_cj = []
            for r in self.possible_rel_values:
                r = to_etype_name(r)
                drug_ci.append(graph['rev-%s' % r].in_degrees())
                disease_ci.append(graph[r].in_degrees())
                if self._symm:
                    drug_cj.append(graph[r].out_degrees())
                    disease_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    drug_cj.append(th.zeros((self.num_drug,)))
                    disease_cj.append(th.zeros((self.num_disease,)))

            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self.num_drug, )
                disease_cj = th.ones(self.num_disease, )
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                     vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug

    

class DrugDataLoader(object):
    def __init__(self,
                 name,
                 device,
                 symm=True,
                 k=2):
        self._name = name
        self._device = device
        self._symm = symm
        self.num_neighbor = k
        print("Starting processing {} ...".format(self._name))
        self._dir = os.path.join(_paths[self._name])
        self.cv_data_dict = self._load_drug_data(self._dir, self._name)

        self._generate_topoy_graph()
        self.drug_graph, self.disease_graph = self._generate_feat_graph()
        self._generate_feat()
        self.possible_rel_values = self.values

    def _load_drug_data(self, file_path, data_name):
        association_matrix = None
        if data_name in ['Gdataset', 'Cdataset','newGdataset']:
            data = sio.loadmat(file_path)
            association_matrix = data['didr'].T
            self.disease_sim_features = data['fordisease']
            self.drug_sim_features = data['fordrug']
        # elif data_name in ['Ldataset']:
        #     association_matrix = np.loadtxt(os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
        #     self.disease_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
        #     self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
        elif data_name in ['Ldataset','twoLdataset','newLdataset']:
                if data_name == 'twoLdataset' :
                    if not os.path.exists(file_path):
                        path='/home/liujin/data/DRGCL-main/DRGCL-main/raw_data/drug_data/Ldataset/newLdataset.mat'
                        self.generate_twodrug_twodisease(path, file_path,drug_n=12,disease_n=19,fordrug_m=19,fordisease_m=15)
                    
                    # 如果文件已存在，仍然需要加载数据
                    data = sio.loadmat(file_path)
                    association_matrix = data['didr']
                    self.disease_sim_features = data['twodisease']
                    self.drug_sim_features = data['twodrug']
                else:  
                    association_matrix = np.loadtxt(os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
                    self.disease_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
                    self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
                    
        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            association_matrix = data.values
            self.disease_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values

        # self.disease_sim_features = th.FloatTensor(self.disease_sim_features)
        # self.drug_sim_features = th.FloatTensor(self.drug_sim_features)

        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]

        kfold = KFold(n_splits=10, shuffle=True, random_state=1024)
        pos_row, pos_col = np.nonzero(association_matrix)
        neg_row, neg_col = np.nonzero(1 - association_matrix)
        assert len(pos_row) + len(neg_row) == np.prod(association_matrix.shape)
        cv_num = 0
        cv_data = {}
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                kfold.split(neg_row)):
            train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
            train_pos_values = [1] * len(train_pos_edge[0])
            train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
            train_neg_values = [0] * len(train_neg_edge[0])

            test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
            test_pos_values = [1] * len(test_pos_edge[0])

            '''
            # test positive and test negative ration is 1:1
                test_neg_edge = np.stack([neg_row[test_neg_idx][0:len(test_pos_values)],
                                          neg_col[test_neg_idx][0:len(test_pos_values)]])

            '''

            test_neg_edge = np.stack([neg_row[test_neg_idx],
                                      neg_col[test_neg_idx]])
            test_neg_values = [0] * len(test_neg_edge[0])

            train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
            train_values = np.concatenate([train_pos_values, train_neg_values])
            test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
            test_values = np.concatenate([test_pos_values, test_neg_values])

            train_data = {
                'disease_id': train_edge[0],
                'drug_id': train_edge[1],
                'values': train_values
            }
            train_data_info = pd.DataFrame(train_data, index=None)

            test_data = {
                'disease_id': test_edge[0],
                'drug_id': test_edge[1],
                'values': test_values
            }
            test_data_info = pd.DataFrame(test_data, index=None)
            values = np.unique(train_values)
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data

    def _generate_feat(self):
        self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3)
        self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3)

        self.drug_feature = th.cat(
            [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
             th.zeros([self.num_drug, 1])], 1)

        self.disease_feature = th.cat(
            [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
             th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1)

    def _generate_topoy_graph(self):
        self.data_cv = {}
        for cv in range(0, 10):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values

            train_pairs, train_values = self._generate_pair_value(
                self.train_rel_info)
            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)

            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv

    def _generate_feat_graph(self):
        # drug feature graph
        drug_sim = self.drug_sim_features
        drug_num_neighbor = self.num_neighbor
        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]

        drug_neighbor = np.argpartition(-drug_sim, kth=drug_num_neighbor, axis=1)[:, :drug_num_neighbor]
        dr_row_index = np.arange(drug_neighbor.shape[0]).repeat(drug_neighbor.shape[1])
        dr_col_index = drug_neighbor.reshape(-1)
        drug_edge_index = np.array([dr_row_index, dr_col_index]).astype(int).T

        drug_edges = np.array(list(drug_edge_index), dtype=np.int32).reshape(drug_edge_index.shape)
        drug_adj = sp.coo_matrix((np.ones(drug_edges.shape[0]), (drug_edges[:, 0], drug_edges[:, 1])),
                                 shape=(self.num_drug, self.num_drug),
                                 dtype=np.float32)
        drug_adj = drug_adj + drug_adj.T.multiply(drug_adj.T > drug_adj) - drug_adj.multiply(
            drug_adj.T > drug_adj)
        # drug_graph = normalize(drug_adj + sp.eye(drug_adj.shape[0]))
        drug_graph = normalize(drug_adj)
        drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)
        # disease feature graph
        disease_sim = self.disease_sim_features
        disease_num_neighbor = self.num_neighbor
        if disease_num_neighbor > disease_sim.shape[0] or disease_num_neighbor < 0:
            disease_num_neighbor = disease_sim.shape[0]

        disease_neighbor = np.argpartition(-disease_sim, kth=disease_num_neighbor, axis=1)[:, :disease_num_neighbor]
        di_row_index = np.arange(disease_neighbor.shape[0]).repeat(disease_neighbor.shape[1])
        di_col_index = disease_neighbor.reshape(-1)
        disease_edge_index = np.array([di_row_index, di_col_index]).astype(int).T

        disease_edges = np.array(list(disease_edge_index), dtype=np.int32).reshape(disease_edge_index.shape)
        disease_adj = sp.coo_matrix((np.ones(disease_edges.shape[0]), (disease_edges[:, 0], disease_edges[:, 1])),
                                    shape=(self.num_disease, self.num_disease),
                                    dtype=np.float32)
        disease_adj = disease_adj + disease_adj.T.multiply(disease_adj.T > disease_adj) - disease_adj.multiply(
            disease_adj.T > disease_adj)
        # disease_graph = normalize(disease_adj + sp.eye(disease_adj.shape[0]))
        disease_graph = normalize(disease_adj)
        disease_graph = sparse_mx_to_torch_sparse_tensor(disease_graph)

        return drug_graph, disease_graph

    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["disease_id"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["drug_id"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rel_values:
            ridx = np.where(
                rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('drug', str(rating), 'disease'): (rrow, rcol),
                ('disease', 'rev-%s' % str(rating), 'drug'): (rcol, rrow)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            drug_ci = []
            drug_cj = []
            disease_ci = []
            disease_cj = []
            for r in self.possible_rel_values:
                r = to_etype_name(r)
                drug_ci.append(graph['rev-%s' % r].in_degrees())
                disease_ci.append(graph[r].in_degrees())
                if self._symm:
                    drug_cj.append(graph[r].out_degrees())
                    disease_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    drug_cj.append(th.zeros((self.num_drug,)))
                    disease_cj.append(th.zeros((self.num_disease,)))

            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self.num_drug, )
                disease_cj = th.ones(self.num_disease, )
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                     vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug

    
import collections

class yvDrugDataLoader(object):
    def __init__(self,
                 name,
                 device,
                 symm=True,
                 top_percent=0.03):  # 改为top_percent参数
        self._name = name
        self._device = device
        self._symm = symm
        self.top_percent = top_percent  # 保存参数
        print("Starting processing {} ...".format(self._name))
        self._dir = os.path.join(_paths[self._name])
        self.cv_data_dict = self._load_drug_data(self._dir, self._name)

        self._generate_topoy_graph()
        self.drug_graph, self.disease_graph = self._generate_feat_graph(self.top_percent)
        self._generate_feat()
        self.possible_rel_values = self.values


    def _load_drug_data(self, file_path, data_name):
        association_matrix = None
        if data_name in ['Gdataset', 'Cdataset']:
            data = sio.loadmat(file_path)
            association_matrix = data['didr'].T
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']
        elif data_name in ['Ldataset']:
            association_matrix = np.loadtxt(os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
            self.disease_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
            self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            association_matrix = data.values
            self.disease_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values

        # self.disease_sim_features = th.FloatTensor(self.disease_sim_features)
        # self.drug_sim_features = th.FloatTensor(self.drug_sim_features)

        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]

        kfold = KFold(n_splits=10, shuffle=True, random_state=1024)
        pos_row, pos_col = np.nonzero(association_matrix)
        neg_row, neg_col = np.nonzero(1 - association_matrix)
        assert len(pos_row) + len(neg_row) == np.prod(association_matrix.shape)
        cv_num = 0
        cv_data = {}
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                kfold.split(neg_row)):
            train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
            train_pos_values = [1] * len(train_pos_edge[0])
            train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
            train_neg_values = [0] * len(train_neg_edge[0])

            test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
            test_pos_values = [1] * len(test_pos_edge[0])

            '''
            # test positive and test negative ration is 1:1
                test_neg_edge = np.stack([neg_row[test_neg_idx][0:len(test_pos_values)],
                                          neg_col[test_neg_idx][0:len(test_pos_values)]])

            '''

            test_neg_edge = np.stack([neg_row[test_neg_idx],
                                      neg_col[test_neg_idx]])
            test_neg_values = [0] * len(test_neg_edge[0])

            train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
            train_values = np.concatenate([train_pos_values, train_neg_values])
            test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
            test_values = np.concatenate([test_pos_values, test_neg_values])

            train_data = {
                'disease_id': train_edge[0],
                'drug_id': train_edge[1],
                'values': train_values
            }
            train_data_info = pd.DataFrame(train_data, index=None)

            test_data = {
                'disease_id': test_edge[0],
                'drug_id': test_edge[1],
                'values': test_values
            }
            test_data_info = pd.DataFrame(test_data, index=None)
            values = np.unique(train_values)
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data

    def _generate_feat(self):
        self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3)
        self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3)

        self.drug_feature = th.cat(
            [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
             th.zeros([self.num_drug, 1])], 1)

        self.disease_feature = th.cat(
            [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
             th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1)

    def _generate_topoy_graph(self):
        self.data_cv = {}
        for cv in range(0, 10):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values

            train_pairs, train_values = self._generate_pair_value(
                self.train_rel_info)
            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)

            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv

    def _generate_feat_graph(self, top_percent=0.03):
        # 处理drug邻居
        drug_sim = self.drug_sim_features.copy()
        drug_num = drug_sim.shape[0]

        np.fill_diagonal(drug_sim, 0)
        sim_flat = drug_sim.flatten()
        threshold = np.percentile(sim_flat, 100 * (1 - top_percent))
        drug_row, drug_col = np.where(drug_sim >= threshold)

        for i in range(drug_num):
            neighbors = drug_col[drug_row == i]
            if len(neighbors) == 0:
                sim_i = drug_sim[i, :]
                sim_i[i] = 0
                max_idx = np.argmax(sim_i)
                drug_row = np.append(drug_row, i)
                drug_col = np.append(drug_col, max_idx)

        drug_edges = np.array([drug_row, drug_col]).T

        drug_adj = sp.coo_matrix((np.ones(len(drug_edges)), (drug_edges[:, 0], drug_edges[:, 1])),
                                shape=(drug_num, drug_num), dtype=np.float32)
        drug_adj = drug_adj + drug_adj.T.multiply(drug_adj.T > drug_adj) - drug_adj.multiply(drug_adj.T > drug_adj)
        drug_graph = normalize(drug_adj)
        drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)

        # 处理disease邻居，类似
        disease_sim = self.disease_sim_features.copy()
        disease_num = disease_sim.shape[0]

        np.fill_diagonal(disease_sim, 0)
        sim_flat = disease_sim.flatten()
        threshold = np.percentile(sim_flat, 100 * (1 - top_percent))
        disease_row, disease_col = np.where(disease_sim >= threshold)

        for i in range(disease_num):
            neighbors = disease_col[disease_row == i]
            if len(neighbors) == 0:
                sim_i = disease_sim[i, :]
                sim_i[i] = 0
                max_idx = np.argmax(sim_i)
                disease_row = np.append(disease_row, i)
                disease_col = np.append(disease_col, max_idx)

        disease_edges = np.array([disease_row, disease_col]).T

        disease_adj = sp.coo_matrix((np.ones(len(disease_edges)), (disease_edges[:, 0], disease_edges[:, 1])),
                                    shape=(disease_num, disease_num), dtype=np.float32)
        disease_adj = disease_adj + disease_adj.T.multiply(disease_adj.T > disease_adj) - disease_adj.multiply(disease_adj.T > disease_adj)
        disease_graph = normalize(disease_adj)
        disease_graph = sparse_mx_to_torch_sparse_tensor(disease_graph)

        # 统计邻居数并打印
        self.print_neighbor_stats(drug_graph, "Drug Graph")
        self.print_neighbor_stats(disease_graph, "Disease Graph")

        return drug_graph, disease_graph

    def print_neighbor_stats(self, sparse_tensor, graph_name):
        """
        统计邻居数，并打印最大、最小、平均值及分布
        sparse_tensor: torch.sparse.FloatTensor 格式的邻接矩阵
        """
        # sparse_tensor 是稀疏矩阵，转成稠密度太大不推荐
        # 统计每行非零元素数即邻居数
        # torch.sparse.FloatTensor 格式：(indices, values, size)
        indices = sparse_tensor._indices()  # shape (2, num_edges)
        row_indices = indices[0].tolist()

        counter = collections.Counter(row_indices)
        neighbor_counts = [counter[i] for i in range(sparse_tensor.size(0))]

        max_count = max(neighbor_counts)
        min_count = min(neighbor_counts)
        avg_count = sum(neighbor_counts) / len(neighbor_counts)

        print(f"--- {graph_name} 邻居数统计 ---")
        print(f"节点总数: {sparse_tensor.size(0)}")
        print(f"最大邻居数: {max_count}")
        print(f"最小邻居数: {min_count}")
        print(f"平均邻居数: {avg_count:.2f}")

        # 打印邻居数分布
        freq = collections.Counter(neighbor_counts)
        print("邻居数分布:")
        for neighbor_num, count in sorted(freq.items()):
            print(f"邻居数={neighbor_num} 的节点数={count}")
        print("------------------------")

    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["disease_id"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["drug_id"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rel_values:
            ridx = np.where(
                rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('drug', str(rating), 'disease'): (rrow, rcol),
                ('disease', 'rev-%s' % str(rating), 'drug'): (rcol, rrow)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            drug_ci = []
            drug_cj = []
            disease_ci = []
            disease_cj = []
            for r in self.possible_rel_values:
                r = to_etype_name(r)
                drug_ci.append(graph['rev-%s' % r].in_degrees())
                disease_ci.append(graph[r].in_degrees())
                if self._symm:
                    drug_cj.append(graph[r].out_degrees())
                    disease_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    drug_cj.append(th.zeros((self.num_drug,)))
                    disease_cj.append(th.zeros((self.num_disease,)))

            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self.num_drug, )
                disease_cj = th.ones(self.num_disease, )
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                     vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug

class DrugModeloader(object):
    def __init__(self,
                 name,
                 device,
                 symm=True,
                 k=2):
        self._name = name
        self._device = device
        self._symm = symm
        self.num_neighbor = k
        print("Starting processing {} ...".format(self._name))
        self._dir = os.path.join(_paths[self._name])
        self.cv_data_dict = self._load_drug_data(self._dir, self._name)

        self._generate_topoy_graph()
        self.drug_graph, self.disease_graph = self._generate_feat_graph()
        self._generate_feat()
        self.possible_rel_values = self.values

    def _load_drug_data(self, file_path, data_name):
        association_matrix = None
        if data_name in ['Gdataset', 'Cdataset']:
            data = sio.loadmat(file_path)
            association_matrix = data['didr'].T
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']
            # self.row_idx = [ele for ele in range(0, association_matrix.shape[0])]
        elif data_name in ['Ldataset']:
            association_matrix = np.loadtxt(os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
            self.disease_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
            self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
            # self.row_idx = [ele for ele in range(0, association_matrix.shape[0])]
        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            association_matrix = data.values
            self.disease_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values
            # self.row_idx = [ele for ele in range(0, association_matrix.shape[0])]

        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]
        self.col_idx = [ele for ele in range(0, association_matrix.shape[1])]

        cv_num = 0
        cv_data = {}
        for col in self.col_idx:
            train_matrix = association_matrix.copy()
            test_value = train_matrix[:, col]
            test_data = {
                'drug_id': [idx for idx in range(0, self._num_drug)],
                'disease_id': [col] * len(test_value),
                'values': test_value
            }
            test_data_info = pd.DataFrame(test_data, index=None)

            # train_matrix[:, col] = 0
            pos_row, pos_col = np.nonzero(train_matrix)
            neg_row, neg_col = np.nonzero(1 - train_matrix)

            train_drug_idx = np.hstack([pos_row, neg_row])
            train_disease_idx = np.hstack([pos_col, neg_col])

            pos_values = [1] * len(pos_row)
            neg_values = [0] * len(neg_row)
            train_values = np.hstack([pos_values, neg_values])

            train_data = {
                'drug_id': train_drug_idx,
                'disease_id': train_disease_idx,
                'values': train_values
            }
            train_data_info = pd.DataFrame(train_data, index=None)

            values = np.unique(train_values)
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data

    def _generate_feat(self):
        self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3)
        self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3)

        self.drug_feature = th.cat(
            [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
             th.zeros([self.num_drug, 1])], 1)

        self.disease_feature = th.cat(
            [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
             th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1)

    def _generate_topoy_graph(self):
        self.data_cv = {}
        for cv in range(0, len(self.col_idx)):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values

            train_pairs, train_values = self._generate_pair_value(self.train_rel_info)

            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)

            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv

    def _generate_feat_graph(self):
        # drug feature graph
        drug_sim = self.drug_sim_features
        drug_num_neighbor = self.num_neighbor
        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]

        drug_neighbor = np.argpartition(-drug_sim, kth=drug_num_neighbor, axis=1)[:, :drug_num_neighbor]
        dr_row_index = np.arange(drug_neighbor.shape[0]).repeat(drug_neighbor.shape[1])
        dr_col_index = drug_neighbor.reshape(-1)
        drug_edge_index = np.array([dr_row_index, dr_col_index]).astype(int).T

        drug_edges = np.array(list(drug_edge_index), dtype=np.int32).reshape(drug_edge_index.shape)
        drug_adj = sp.coo_matrix((np.ones(drug_edges.shape[0]), (drug_edges[:, 0], drug_edges[:, 1])),
                                 shape=(self.num_drug, self.num_drug),
                                 dtype=np.float32)
        drug_adj = drug_adj + drug_adj.T.multiply(drug_adj.T > drug_adj) - drug_adj.multiply(
            drug_adj.T > drug_adj)
        drug_graph = normalize(drug_adj + sp.eye(drug_adj.shape[0]))
        drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)
        # disease feature graph
        disease_sim = self.disease_sim_features
        disease_num_neighbor = self.num_neighbor
        if disease_num_neighbor > disease_sim.shape[0] or disease_num_neighbor < 0:
            disease_num_neighbor = disease_sim.shape[0]

        disease_neighbor = np.argpartition(-disease_sim, kth=disease_num_neighbor, axis=1)[:, :disease_num_neighbor]
        di_row_index = np.arange(disease_neighbor.shape[0]).repeat(disease_neighbor.shape[1])
        di_col_index = disease_neighbor.reshape(-1)
        disease_edge_index = np.array([di_row_index, di_col_index]).astype(int).T

        disease_edges = np.array(list(disease_edge_index), dtype=np.int32).reshape(disease_edge_index.shape)
        disease_adj = sp.coo_matrix((np.ones(disease_edges.shape[0]), (disease_edges[:, 0], disease_edges[:, 1])),
                                    shape=(self.num_disease, self.num_disease),
                                    dtype=np.float32)
        disease_adj = disease_adj + disease_adj.T.multiply(disease_adj.T > disease_adj) - disease_adj.multiply(
            disease_adj.T > disease_adj)
        disease_graph = normalize(disease_adj + sp.eye(disease_adj.shape[0]))
        disease_graph = sparse_mx_to_torch_sparse_tensor(disease_graph)

        return drug_graph, disease_graph

    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["drug_id"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["disease_id"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rel_values:
            ridx = np.where(
                rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('drug', str(rating), 'disease'): (rrow, rcol),
                ('disease', 'rev-%s' % str(rating), 'drug'): (rcol, rrow)
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            drug_ci = []
            drug_cj = []
            disease_ci = []
            disease_cj = []
            for r in self.possible_rel_values:
                r = to_etype_name(r)
                drug_ci.append(graph['rev-%s' % r].in_degrees())
                disease_ci.append(graph[r].in_degrees())
                if self._symm:
                    drug_cj.append(graph[r].out_degrees())
                    disease_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    drug_cj.append(th.zeros((self.num_drug,)))
                    disease_cj.append(th.zeros((self.num_disease,)))

            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self.num_drug, )
                disease_cj = th.ones(self.num_disease, )
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                     vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug


if __name__ == '__main__':
    # DrugDataLoader("lrssl", device=th.device('cpu'), symm=True)
    # DrugNovoLoader("Gdataset", device=th.device('cpu'), symm=True)
    DiseaseNovoLoader("Gdataset", device=th.device('cpu'), symm=True)