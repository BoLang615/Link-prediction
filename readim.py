import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from preprocessing import sparse_to_tuple
import torch
import os

#from sklearn import preprocessing
from sklearn.preprocessing import normalize
#from matplotlib import pyplot as plt
#from sklearn.metrics import roc_auc_score, average_precision_score


#from input_data import load_data

import collections

import cv2

from matplotlib import colors

file_path = 'connection.txt'
coor_file_path = 'chairs_coor.txt'
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_dict(dict_graph):
    '''
    dict_graph : dictionary
        The graph in the dict format
    Returns
    -------
    adj : scipy.sparse.csr.csr_matrix
        The adjacent matrix of the graph
    graph : collections.defaultdict
    '''
    
    graph = collections.defaultdict(list)
    for key, value in dict_graph.items():
        for item in value:
            graph[key].append(item)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    return adj, graph

def load_data_txt(file_path):
    '''
    file_path : str
        The path of the txt file
    Returns
    -------
    adj : scipy.sparse.csr.csr_matrix
        The adjacent matrix of the graph
    graph : collections.defaultdict
    '''
    file = open(file_path, 'r') 
    dict_temp = {}
    while True: 
        line = file.readline() 
        # if line is empty 
        # end of file is reached 
        if not line: 
            break
        key = int(line.split(":")[0])
        dict_temp[key] = []
        for item in line.split(":")[1].split(','):
            
            dict_temp[key].append(int(item))      
    file.close()
    adj, graph = load_data_dict(dict_temp)
    return adj, graph

#adj_temp, graph_temp = load_data_txt(file_path)

def load_all_coor(coor_file_path):
    '''
    coor_file_path : path
        Format in txt: name p0_coor p1_coor ...
            1.jpg 433,252 234,433 
            2.jpg 12,34 23,43 
            # At each line, there exits a space.

    Return:
        all_coor_dict
            {'1.jpg': [[433, 252], [234, 433]], '2.jpg': [[12, 34], [23, 43]]}

    '''
    file = open(coor_file_path, 'r') 
    all_coor_dict = {}
    while True: 
        line = file.readline() 
        # if line is empty 
        # end of file is reached 
        if not line: 
            break
        context = line.split(' ')
        if context[0].split('.')[-1]=='jpg':
            all_coor_dict[context[0]] = []
            for item in context[1:-1]:
                term = []
                x, y = item.split(',')
                term.append(int(x))
                term.append(int(y))
                all_coor_dict[context[0]].append(term)
    file.close()
    return(all_coor_dict)

def adj2data(csr_matrix):
    '''

    Parameters
    ----------
    csr_matrix : csr_matrix
        get the index of row, col and data from the csr_matrix
    Returns
    -------
    row_arr : array
    col_arr : array
    data_arr : array
    '''
    row_arr = csr_matrix.tocoo().row
    col_arr = csr_matrix.tocoo().col
    data_arr = csr_matrix.toarray()
    return row_arr, col_arr, data_arr

def adj2connection(csr_matrix):
    '''
    Get the connection of each node
    Return is an array
    '''
    con = []
    for key in csr_matrix.todok().keys():
        temp = np.asarray([key[1], key[0]])
        con.append(temp)
    return np.asarray(con)




def show_connection(img_name, coor, con):
    '''
    Read the connection, coordinates and one image,
    draw the nodes and the connload_data_dict(dict_graph)ection on the image.
    '''
    
    background = np.zeros((720,1280,3), np.uint8)
    background += 255
        
    row_max, col_max = np.max(coor,axis=0)
    if row_max >= 720 or col_max >= 1280:
        for item in coor:
            item[0] /= 2
            item[1] /= 2
    
    color_list = ['hotpink', 'midnightblue', 'navy', 'plum', 'seagreen', 'black', 'purple', 'tan', 'wheat', 'chocolate']
    count = 0
    for point in coor:
        t = colors.hex2color(colors.cnames[color_list[count]])
        color = (int(t[0]*255), int(t[1]*255), int(t[2]*255))
        cv2.circle(background, (int(point[0]),int(point[1])), 5, color, (-1))
        count += 1
        
    for count in np.arange(con.shape[0]):
        cv2.line(background, (int(coor[con[count][0]][0]), int(coor[con[count][0]][1])), (int(coor[con[count][1]][0]), int(coor[con[count][1]][1])), (0,0,255), (1))
        
    
    cv2.imshow(img_name, background)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features, graph

# def get_scores(edges_pos, edges_neg, adj_rec):

#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))

#     # Predict on test set of edges
#     preds = []
#     pos = []
#     for e in edges_pos:
#         # print(e)
#         # print(adj_rec[e[0], e[1]])
#         preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
#         pos.append(adj_orig[e[0], e[1]])

#     preds_neg = []
#     neg = []
#     for e in edges_neg:

#         preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
#         neg.append(adj_orig[e[0], e[1]])

#     preds_all = np.hstack([preds, preds_neg])
#     labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
#     roc_score = roc_auc_score(labels_all, preds_all)
#     ap_score = average_precision_score(labels_all, preds_all)

#     return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

def preprocess_graph(adj):
    '''
    normalize adj
    '''
    adj = sp.coo_matrix(adj) #确保adj为csr_matrix
    #在计算新特征时没有考虑自己的特征，这肯定是个重大缺陷,so在adj上给对角线元素全部加1
    adj_ = adj + sp.eye(adj.shape[0])#在adj上给对角线元素全部加1
    rowsum = np.array(adj_.sum(1)) #按照行来计算和
    #np.power(rowsum, -0.5).flatten():将rowsum的元素进行开方后，拉平一行
    #sp.diags https://www.cnblogs.com/SupremeBoy/p/12952735.html
    # degree_mat_inv_sqrt的结果是把（）算好的值填写到10×10的主对角线上去。其余位置补0
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    #.tocoo():Convert this matrix to COOrdinate format
    #normalize adj: 采用加法规则进行聚合，对于度大的节点特征越来越大，而对于度小的节点却相反，这可能导致网络训练过程中梯度爆炸或者消失的问题。
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def get_features(feat_shape, img_name, all_coor_dict):
    #temp = np.zeros((feat_shape[0], feat_shape[1]))
    
    image = all_coor_dict[img_name]
    
    norm2 = normalize(np.asarray(image), axis=0)
    features = sp.lil_matrix(norm2)

    return features

def data_process(adj):
    
    adj_norm = preprocess_graph(adj)
    
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

