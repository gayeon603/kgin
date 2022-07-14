import math
import os
import pickle
from tqdm import tqdm
import random
import copy

import torch
import torch.nn as nn
import gensim
import faiss
import gc

# from kmeans_pytorch import kmeans
import time



class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id, device):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = gpu_id
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []
        gc.collect()
        torch.cuda.empty_cache()

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        # print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster) #(64,256)
        clus.verbose = verbose
        clus.niter = niter # 클러스터링 횟수
        clus.nredo = nredo #클러스터링 재진행
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        index=faiss.IndexFlatL2(hidden_size)
        
        torch.cuda.empty_cache()
        
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster: # 클러스터 개수보다 데이터 개수가 더 많으면 hidden size 차원만큼 인덱스 설정해서 학습 
            self.clus.train(x, self.index)
        # get cluster centroids
        
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size) #(4,40)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)
        
        torch.cuda.empty_cache()

    def query(self, x):
        print('query starts')
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        #print('distance',D)
        #print('index', I)
        seq2cluster = [int(n[0]) for n in I]
        print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        print('centroids',self.centroids[seq2cluster].shape)
        print('centroids',self.centroids[seq2cluster][0])
        return seq2cluster, self.centroids[seq2cluster]

