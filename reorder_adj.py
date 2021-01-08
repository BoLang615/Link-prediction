import readim as rd
import numpy as np
import args

import collections
import networkx as nx
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt




def reorder_single_coor(n_point, single_coor_dict):
    '''
    Reorder the coordinate of one coor_dict
    ----------
    Parameters:
    ----------    
    n_point : int
        number of nodes
    single_coor_dict : dict
        {'00001241.jpg': [[54, 410], [250, 484], [316, 340], [160, 283], [43, 313], [230, 367], [301, 286], [141, 246], [187, 33], [328, 33]]}
    ----------
    Returns:
    ----------
    index_rand : ARRAY
        New random order of the coordinates. Eg: array([1, 3, 7, 5, 8, 6, 0, 4, 9, 2])
    reorder_dict : dict
        {0: 1, 1: 3, 2: 7, 3: 5, 4: 8, 5: 6, 6: 0, 7: 4, 8: 9, 9: 2}
    reorder_all_coor_dict : dict
        The image with a new random order of coordinates.
        {'00001241.jpg': [[250, 484], [160, 283], [141, 246], [230, 367], [187, 33], [301, 286], [54, 410], [43, 313], [328, 33], [316, 340]]}
    '''
    
    
    for key, value in single_coor_dict.items():
        index_rand = np.arange(n_point)
        np.random.shuffle(index_rand)
        reorder_all_coor_dict = {}
        # print("key: {}, value: {}".format(key, value))
        # print(key)
        # print("New order is: {}".format(index_rand))
        reorder_coor = []
        for item in index_rand:
            reorder_coor.append(value[item])
        reorder_all_coor_dict[key] = reorder_coor
        #print(reorder_all_coor_dict)
        reorder_dict = {}
    reorder_dict = {}
    for index in np.arange(0,n_point):
        reorder_dict[index] = index_rand[index]
    #print("The corresponding nodes are: {}".format(reorder_dict))
    return key, index_rand, reorder_dict, reorder_all_coor_dict


def reorder_connection(file_path, reorder_dict):
    '''
    Get the new order of the connection
    ----------
    Parameters:
    ---------- 
    file_path : str
        The path of the connection.txt
    ----------
    Returns:
    ----------
    reorder_connection_dict : dict
            New order of the connection

    '''
    file = open(file_path, 'r') 
    reorder_connection_dict = {}
    while True: 
        line = file.readline() 
        # if line is empty 
        # end of file is reached 
        if not line: 
            break
        key = int(line.split(":")[0])
        reorder_key = int(reorder_dict[key])
        reorder_connection_dict[reorder_key] = []
        for item in line.split(":")[1].split(','):
            reorder_connection_dict[reorder_key].append(reorder_dict[int(item)])
    file.close()
    return reorder_connection_dict

def reorder_graph(reorder_connection_dict, show = False):
    '''
    To build the graph with new order of nodes
    and get the new adjancy matrix

    Parameters
    ----------
    reorder_connection_dict : dict
        New order of the connection
    show : BOol, optional
        show the new graph

    Returns
    -------
    adj_reorder : csr_matrix
        thr adj csr_matrix.
    graph_new : networkx.classes.graph.Graph
        the graph with new order of nodes

    '''
    graph_new_list = collections.defaultdict(list)
    for key, value in reorder_connection_dict.items():
        for item in value:
            graph_new_list[key].append(item)
    
    graph_new = nx.from_dict_of_lists(graph_new_list)
    
    adj_reorder_data = np.zeros((args.n_point, args.n_point), dtype = int)
    for key, value in reorder_connection_dict.items():
        row = key
        for item in value:
            col = value
            adj_reorder_data[row, col] = 1
    
    adj_reorder = csr_matrix(adj_reorder_data)
    
    if show:
        nx.draw_networkx(graph_new)
        plt.show()
    
    return adj_reorder, graph_new

def reorder_all(all_coor_dict, n_point, ori_cconnection_file_path, show = False):
    reorder_adj_all_dict = {}
    reorder_all_coor_dict = {}
    for item in all_coor_dict.items():
        key, index_rand, reorder_dict, reorder_coor_dict = reorder_single_coor(n_point, {item[0]: item[1]})
        reorder_connection_dict = reorder_connection(ori_cconnection_file_path, reorder_dict)
        adj_reorder, graph_new = reorder_graph(reorder_connection_dict, show = show)
        reorder_adj_all_dict[key] = adj_reorder
        reorder_all_coor_dict[key] = reorder_coor_dict[key]
    return reorder_all_coor_dict, reorder_adj_all_dict
    
# all_coor_dict = rd.load_all_coor(args.coor_file_path_test)
# reorder_all_coor_dict, reorder_adj_all_dict = reorder_all(all_coor_dict, args.n_point, ori_cconnection_file_path = args.file_path, show = False)

    
# a = getattr(rd, 'load_data_txt')(args.file_path)
# print(a)

# 我检查了一下我自己写的adj生成和nx.adjacency_matrix的结果是一样的
# connection_dict = reorder_connection(args.file_path, range(10))
# adj_reorder, graph_new = reorder_graph(connection_dict, show = False)


