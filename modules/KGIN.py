'''
Created on July 1, 2020
PyTorch Implementation of KGIN
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

from builtins import print
from errno import EDEADLK
from lib2to3.pgen2.token import NEWLINE
import random
from time import time
from openpyxl import NUMPY
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch_scatter import scatter_mean
from modules.intent_cluster import KMeans
from sklearn.decomposition import PCA
import gc
import faiss
import GPUtil
from threading import Thread
import time

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        
#monitor = Monitor(10)
#monitor.stop()



class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph_AK, adj_mat,seed):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.n_epoch = data_config['epoch_num']
        self.r_num0 = data_config['num_r0']

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.emb_size1 = args_config.dim1
        self.k_att = args_config.k_att
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")
        self.gpu_id = args_config.gpu_id
        self.seed=seed

        cs1 = []
        for i in range(self.n_relations):
            cs1.append(40)
        self.dim_f = int(np.sum(cs1))
        self.dim_flag = np.array(cs1, dtype=np.int).tolist()

        self.dim_flag1 = [0]
        for i in range(self.n_relations):
            self.dim_flag1.append(self.dim_flag1[-1] + self.dim_flag[i])

        self.adj_mat = adj_mat
        self.graph_AK = graph_AK

        self.cluster=KMeans(
                    num_cluster=self.n_factors,
                    seed=self.seed,
                    hidden_size= 40, 
                    gpu_id=self.gpu_id,
                    device=self.device,
                )

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)
        self.all_embed_ak = nn.Parameter(self.all_embed_ak)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        gain = 1.414

        self.all_embed_ak = initializer(torch.empty(self.n_entities, self.dim_f), gain=gain)
        self.all_embed = initializer(torch.empty(self.n_users, self.dim_f))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.dim_f))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        
        self.W_R = []
        for i in range(self.n_relations):
            a = torch.zeros(1, self.dim_f)
            a[0, self.dim_flag1[i]:self.dim_flag1[i + 1]] += 1
            self.W_R.append(a)
        self.W_R = torch.cat(self.W_R, dim=0).to(self.device)


    def _init_model(self):
        return GraphConv(channel=self.dim_f,
                         emb_size=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items=self.n_items,
                         n_entities=self.n_entities,
                         n_relations=self.n_relations,
                         n_factors=self.n_factors,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                         dim_flag=self.dim_flag,
                         r_num0=self.r_num0,
                         k_att=self.k_att,
                         device=self.device,
                         gpu_id=self.gpu_id,
                         cluster=self.cluster)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]


    def forward(self, batch, graph_AK, graph_UIS):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed
        item_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     self.latent_emb,
                                                     self.interact_mat,
                                                     graph_AK, graph_UIS, 
                                                     self.all_embed_ak,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)        

        u_e = user_gcn_emb[user]
        pos_e, neg_e = item_gcn_emb[pos_item], item_gcn_emb[neg_item]

        return self.create_bpr_loss(u_e, pos_e, neg_e, cor)

    def generate(self, graph, graph_UIS):
        user_emb = self.all_embed
        return self.gcn(user_emb,
                        self.latent_emb,
                        self.interact_mat,
                        graph, graph_UIS, 
                        self.all_embed_ak,
                        mess_dropout=False, node_dropout=False)[:-1]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items, cor):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss + cor_loss, mf_loss, emb_loss, cor


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, emb_size, n_hops, n_users, n_items, n_entities, 
                 n_relations, n_factors, interact_mat,
                 ind, node_dropout_rate, mess_dropout_rate, 
                 dim_flag, r_num0, k_att, device, gpu_id, cluster) :

        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.channel = channel
        self.emb_size = emb_size
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.n_hops = n_hops
        self.n_items = n_items
        self.n_entities = n_entities

        self.dim_flag = dim_flag
        self.r_num0 = r_num0
        self.k_att = k_att

        self.temperature = 0.2
        
        self.device = device
        self.gpu_id = gpu_id
        initializer = nn.init.xavier_uniform_
        
        self.cluster=cluster
        # weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        # self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        # disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))
        #self.disen_weight_att = nn.Parameter(disen_weight_att)

        #AKGAN의 item 임베딩 먼저 불러 주고
        self.dim_flag1 = [0]
        for i in range(self.n_relations):
            self.dim_flag1.append(self.dim_flag1[-1] + self.dim_flag[i])

        self.convs_ui0 = nn.ModuleList()

        self.tri_embed_u = nn.ModuleList()

        self.tri_embed = Edge_Encoder(n_users=n_users, n_entities=n_entities, n_items=n_items, n_relations=n_relations, 
                                      emb_size=self.emb_size, dim=channel, dim_flag1=self.dim_flag1, device=self.device, gpu_id=self.gpu_id)

        for j in range(n_hops):
            self.convs_ui0.append(
                Neighbor_Ecoder(n_users=n_users, n_entities=n_entities, n_items=n_items,
                     n_relations=n_relations,
                     dim=channel))

        self.dropout = nn.Dropout(p=0.1)  # mess dropout
        
        ################################

        #KGIN의 Aggregator 불러줌 (사용자 임베딩만 넘겨 받기로 수정)
        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_factors=n_factors, device = device))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def _cul_cor(self, cent_att):

        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)
        def MutualInformation():
            # disen_T: [num_factor, dimension]
            #disen_T = centroids.t()
            disen_weight_att=cent_att
            disen_T = disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            #ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, disen_weight_att), dim=1)


            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score


        """cul similarity for each latent factor weight pairs"""
        #centroids = centroids
        disen_weight_att=cent_att
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        #cor += DistanceCorrelation(centroids[i], centroids[j])
                        cor += DistanceCorrelation(disen_weight_att[i], disen_weight_att[j])
                    else:
                        #cor += CosineSimilarity(centroids[i], centroids[j])
                        cor += CosineSimilarity(disen_weight_att[i], disen_weight_att[j])
        return cor

    def forward(self, user_emb, latent_emb, interact_mat, graph_AK, graph_UIS, all_embed_ak,
                mess_dropout=True, node_dropout=False):

        #####################################################
        ##########  AKGAN 아이템 임베딩 구하는 부분 ###########
        entity_embed_ak = all_embed_ak

        if mess_dropout:

            random_indices = np.random.choice(graph_UIS.edges()[0].shape[0],
                                              size=int(graph_UIS.edges()[0].shape[0] * self.mess_dropout_rate),
                                              replace=False)
            graph_UIS = dgl.edge_subgraph(graph_UIS, random_indices, preserve_nodes=True)

        """cal edge embedding"""
        entity_embed0, edge_emb = self.tri_embed(graph_AK, entity_embed_ak)

        entity_res_emb = entity_embed0

        #torch.cuda.empty_cache()
        for j in range(self.n_hops):
            #print(j)
            entity_embed0 = self.convs_ui0[j](graph_AK, entity_embed0)
            
            if mess_dropout:
                entity_embed0 = self.dropout(entity_embed0)
                
            entity_res_emb = torch.add(entity_res_emb, entity_embed0)


        flag = 0
        kmeans_entity_emb = 0
        #print(edge_emb[0].size())
        for i in edge_emb:
            if flag == 0:
                kmeans_entity_emb = i.expand(1, self.n_entities, 40)
                flag = 1
                continue
            kmeans_entity_emb= torch.cat([kmeans_entity_emb, i.expand(1, self.n_entities, 40)],0)

        #print('kmeans_entity_emb size', kmeans_entity_emb.size())

       
        ####################relation별로 의도 클러스터링하기##############################

        #kmeans_start=time()

        cluster=self.cluster
        
        #print('kmeans_entity_emb size', kmeans_entity_emb.size())#[19, 106389, 40]

        new=[]
        for i in kmeans_entity_emb:
            new.append(i[i.sum(dim=1)!=0])

        #del kmeans_entity_emb
        #torch.cuda.empty_cache()
        
        new_train=torch.cat(new,0)
        #new=kmeans_entity_emb[kmeans_entity_emb.sum(dim=2)!=0] #zero vector 제거  
          
        #print('new_train',new_train.size()) #(283615, 40)
        new_train=new_train.detach().cpu().numpy().astype('float32')

        #print('new_train',new_train.shape)
        cluster.train(new_train)
        centroids = cluster.centroids #[4, 40]
        # print(centroids)
        # print(centroids.size())
        #print('centorids',centroids)
        #start=time()

        intent0=[]
        intent1=[]
        intent2=[]
        intent3=[]
        att0=[]
        att1=[]
        att2=[]
        att3=[]

        ##relation별로 각 centroids에 가까운 값  ->0,1 relation은 각각 0,5개 있음
        # ngpus = faiss.get_num_gpus()
        # #print("number of GPUs:", ngpus)
        # for i in new:
        #     index=faiss.IndexFlatL2 (i.shape[1])
        #     index.add(i.detach().cpu().numpy())
        #     if i.shape[0]>0:

        #         D, I = index.search (cluster.centroids.detach().cpu().numpy(), i.shape[0]) #0.2-3
                
        #         intent0.extend(D[0])
        #         intent1.extend(D[1])
        #         intent2.extend(D[2])
        #         intent3.extend(D[3])

        # intent0=torch.Tensor(intent0)
        # intent1=torch.Tensor(intent1)
        # intent2=torch.Tensor(intent2)
        # intent3=torch.Tensor(intent3)


        # output=torch.stack([intent0, intent1, intent2, intent3],dim=0).to(self.device)
   

        # #start=time()
        index=faiss.IndexFlatL2(new_train.shape[1])
        index.add(new_train)
        D, I = index.search (cluster.centroids.detach().cpu().numpy(), 19)

        weight=[[] for _ in range(self.n_factors)]
        
        #start=time()
        for idx,i in enumerate(I):
            for j in i:
                weight[idx].append((index.reconstruct(int(j))))
        #end=time()
        weight=torch.Tensor(weight)
        weight=weight.reshape(4,-1).to(self.device)
        #print(weight.shape)



        #print(intent1)
        #전체 데이터에 대해 한번에 거리찾기 5-6초 걸림,,
        # D,I=cluster.index.search(new_train, 1)
        # print(len(D))
        # print(len(I))

        #relation별로 거리찾기 8초
        # for i in new:
        #         i=i.detach().cpu().numpy().astype('float32')
        #         D,I=cluster.index.search(i, 1)
        #         for idx,d in enumerate(D):
        #             if I[idx]==0:
        #                         intent0.append(d)
        #             elif I[idx]==1:
        #                         intent1.append(d)
        #             elif I[idx]==2:
        #                         intent2.append(d)                        
        #             elif I[idx]==3:
        #                         intent3.append(d)


        
        #각각하면 시간이 너무 오래걸림 7초정도
        # for i in new:
        #     for j in i:
        #         j=j.detach().cpu().numpy().astype('float32')
        #         D,I=cluster.index.search(j.reshape(1,-1), 1)
        #         if I[0][0]==0:
        #             intent0.append(D)
        #         elif I[0][0]==1:
        #             intent1.append(D)
        #         elif I[0][0]==2:
        #             intent2.append(D)
        #         elif I[0][0]==3:
        #             intent3.append(D)

        #end=time()
        #print('kmeans time',end-start)

        # print('intent0',len(intent0))
        # print('intent1',len(intent1))
        # print('intent2',len(intent2))
        # print('intent3',len(intent3))

        """node dropout"""
        if node_dropout:
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        user_res_emb = user_emb  # [n_users, channel]


        # resize = nn.Linear(centroids.size()[1], 760).to(self.device)
        # centor = torch.empty(self.n_factors, 760).to(self.device)

        # for i in range(self.n_factors):
        #     centor[i] = resize(centroids[i])
     
        #print(centor.size())
        #print(centor)
        
        # resize1 = nn.Linear(centor.size()[1], 18).to(self.device)

        # cent_att = torch.empty(self.n_factors, 18).to(self.device)
        # for i in range(self.n_factors):
        #     cent_att[i] = resize1(centor[i])

        #print(cent_att.size())
        #print(weight.size())

        #cor = self._cul_cor(centroids)
        cor = self._cul_cor(centroids)

        #torch.cuda.empty_cache()
        for i in range(len(self.convs)):
            #user_emb = self.convs[i](entity_res_emb, user_emb, latent_emb, centroids, interact_mat)
            user_emb = self.convs[i](entity_res_emb, user_emb, latent_emb, weight, interact_mat)

            """message dropout"""
            if mess_dropout:
                user_emb = self.dropout(user_emb)

            user_emb = F.normalize(user_emb)
            
            """result emb"""
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor
    

class Edge_Encoder(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, emb_size, dim, dim_flag1, device, gpu_id):
        super(Edge_Encoder, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.emb_size = emb_size
        self.dim = dim
        self.n_relations = n_relations
        self.dim_flag1 = dim_flag1
        self.gpu_id=gpu_id
        self.device = device 
        
    def cal_attribute1(self, edges):
        edge_emb = edges.src['node'] * self.W_r
        return {'emb': edge_emb}
    
    def forward(self, graph, entity_emb):
        graph = graph.local_var()
        
        for i in range(self.n_relations):
            graph.nodes['item'].data['node{}'.format(i)] = entity_emb[:, self.dim_flag1[i]:self.dim_flag1[i + 1]]

        edge_emb = []
        for i in range(self.n_relations):
            self.r = i
            """mean_edge"""
            graph.update_all(
                dgl.function.copy_u('node{}'.format(i), 'send'),
                dgl.function.mean('send', 'node0{}'.format(i)), etype=i)

            edge_emb.append(graph.nodes['item'].data['node0{}'.format(i)])

        entity_emb_f=[]
        entity_emb_f = torch.cat(edge_emb, dim=1)

        return entity_emb_f, edge_emb


class Neighbor_Ecoder(nn.Module):

    def __init__(self, n_users, n_entities, n_items, n_relations, dim):
        super(Neighbor_Ecoder, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_items = n_items
        self.dim = dim
        self.n_relations = n_relations

    def cal_attribute1(self, edges):

        edge_emb = edges.src['node'] * self.W_r

        return {'emb': edge_emb}

    def cal_attribute2(self, edges):
        att = edges.data['att'] / edges.dst['nodeatt']
        return {'att1': att}

    def e_mul_e(self, edges):
        att = edges.data['att1'].unsqueeze(1) * edges.data['emb']
        return {'nei': att}

    def forward(self, graph, entity_emb):
        graph = graph.local_var()
        
        graph.nodes['item'].data['node'] = entity_emb

        entity_emb_f = 0
        for i in range(self.n_relations):
            graph.update_all(
                dgl.function.copy_u('node', 'send'),
                dgl.function.mean('send', 'node{}'.format(i)), etype=i)

            entity_emb_f = torch.add(entity_emb_f, graph.nodes['item'].data['node{}'.format(i)])


        return entity_emb_f
    

class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users, n_factors, device):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_factors = n_factors
        self.device = device

    def forward(self, entity_emb, user_emb, latent_emb, weight, interact_mat):

        # n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users
        n_factors = self.n_factors

        """cul user->latent factor attention"""
        score_ = torch.mm(user_emb, latent_emb.t())
        
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_factors, 1]

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        # disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),
        #                         weight).expand(n_users, n_factors, channel)
        
        # resize = nn.Linear(centroids.size()[1], 760).to(self.device)

        # centor = torch.empty(self.n_factors, 760).to(self.device)
        # for i in range(self.n_factors):
        #     centor[i] = resize(centroids[i])


        #print('score',score.size()) #[23566, 4, 1]
        #print('user_agg', user_agg.size()) #[23566, 760]
        disen_weight = weight.expand(n_users, n_factors, channel).to(self.device) 
        # disen_weight = centroids.expand(n_users, n_factors, 40)
        #print('disen',disen_weight.size()) #[23566, 4, 760]

        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]

        return user_agg