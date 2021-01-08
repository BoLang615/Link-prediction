import torch
import torch.nn as nn
import torch.nn.functional as F
#import os
import numpy as np
import scipy.sparse as sp
from preprocessing import sparse_to_tuple

import args
import readim as rd

def tmp_func(x):
    return x

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mish(x):
    return (x*torch.tanh(F.softplus(x)))

def data_process(adj):
    
    adj_norm = rd.preprocess_graph(adj)
    
    adj_train = adj
    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    
    
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    
    
    
    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                                torch.FloatTensor(adj_norm[1]), 
                                torch.Size(adj_norm[2])).to(dev)
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                torch.FloatTensor(adj_label[1]), 
                                torch.Size(adj_label[2])).to(dev)

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(dev)
    weight_tensor[weight_mask] = pos_weight
    return adj_norm, adj_label, norm, weight_tensor


adj_temp, graph = rd.load_data_txt(args.file_path)
adj, adj_label, norm, weight_tensor = data_process(adj_temp)





class VGAE(nn.Module):
	def __init__(self, adj):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj) #relu(WxA)
		#self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x) #WxA...activation=x
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=tmp_func)
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=tmp_func)#WxA...activation=x

	def encode(self, X):#two layers
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)

        
		gaussian_noise = torch.randn(X.size(0), args.hidden2_dim).to(dev)
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z) #sig(Z*ZT)

		return A_pred

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.adj = adj
		self.activation = activation
	#activation(WxA)
	def forward(self, inputs, adj):
		x = inputs
		x = torch.mm(x,self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t())) #sig(Z*ZT)
	return A_pred

def glorot_init(input_dim, output_dim): #initiaze weights by input_dim and output_dim
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)



class GAE(nn.Module):
#	def __init__(self,adj):
	def __init__(self):
		super(GAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, 2 *args.hidden2_dim, adj, activation=mish)
		self.gcn_guess = GraphConvSparse(2 * args.hidden2_dim, args.hidden2_dim, adj, activation=mish)#F.tanh)



	def encode_ori(self, X, adj):
		hidden = self.base_gcn(X, adj)
		z = self.mean = self.gcn_mean(hidden, adj)
		z1 = self.mean = self.gcn_mean(z, adj)
		return z1
    
	def encode_reo(self, X, adj):
		hidden = self.base_gcn(X, adj)
		z = self.mean = self.gcn_mean(hidden, adj)
		z1 = self.mean = self.gcn_mean(z, adj)
		return z1

	def forward(self, X, reorder_X, adj_ori, adj_reorder):
		Z_real = self.encode_ori(X, adj_ori)
		Z_reorder = self.encode_reo(reorder_X, adj_reorder)
		Z = args.alpha * Z_real + args.delta * Z_reorder
		A_pred = dot_product_decode(Z)
#		classify = nn.Linear(args.hidden1_dim, y)
		return A_pred#, classify
    
# class GAE(nn.Module):
# 	def __init__(self,adj):
# 		super(GAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, 2 *args.hidden2_dim, adj, activation=F.tanh)
# 		self.gcn_guess = GraphConvSparse(2 * args.hidden2_dim, args.hidden2_dim, adj, activation=F.tanh)


# 	def encode(self, X):
# 		hidden = self.base_gcn(X)
# 		z = self.mean = self.gcn_mean(hidden)
# 		z1 = self.mean = self.gcn_mean(z)
# 		return z1

# 	def forward(self, X):
# 		Z = self.encode(X)
# 		A_pred = dot_product_decode(Z)
# 		return A_pred
		

# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
# 		return out
