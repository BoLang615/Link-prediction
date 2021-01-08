import torch
from torch_geometric.data import Data, DataLoader
#from torch_geometric.datasets import Planetoid
import readim as rd
import args
import os
from sklearn.preprocessing import normalize
#import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, GCNConv, VGAE
import numpy as np
import scipy.sparse as sp
import shutil


'''
data.x: Node feature matrix with shape [num_nodes, num_node_features]

data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long

data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]

data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]

data.pos: Node position matrix with shape [num_nodes, num_dimensions]
'''



dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feat_shape = (args.n_point, args.num_node_features)

all_coor_dict = rd.load_all_coor(args.coor_file_path)
num_train = len(all_coor_dict) - int(len(all_coor_dict) * args.test_split)
if num_train % args.batch_size != 0:
    num_train = int(num_train / args.batch_size) * args.batch_size - 1
    

adj, graph = rd.load_data_txt(args.file_path)
edge_index = torch.tensor(rd.adj2connection(adj), dtype=torch.long)
row = []
col = []
for item in edge_index.numpy():
    row.append(item[0])
    col.append(item[1])
edge_index_new = torch.tensor([row, col], dtype=torch.long).to(dev)


train_data_list = []
test_data_list = []
test_data_images = {}
count = 0
for key, value in all_coor_dict.items():
    x = torch.tensor(normalize(all_coor_dict[key]), dtype=torch.float).to(dev)
    data = Data(x=x, edge_index=edge_index.t().contiguous())
    if count <= num_train:
        
        train_data_list.append(data)
    else:
        # x = torch.tensor(normalize(all_coor_dict[key]), dtype=torch.float).to(dev)
        # data = Data(x=x, edge_index=edge_index.t().contiguous())
        test_data_list.append(data)
        test_data_images[key] = data
    count += 1
        
train_loader = DataLoader(train_data_list, batch_size = args.batch_size, shuffle=True)
test_loader = DataLoader(test_data_list, shuffle=False)
print("Keys in data are {}.".format(data.keys))
print("Num. of nodes is {}.".format(data.num_nodes))
print("Num. of edges is {}.".format(data.num_edges))
print("Num. of node features is {}.".format(data.num_node_features))




class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        #self.conv3 = GCNConv(3 * out_channels, 2 * out_channels, cached=True)

        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_logstd = GCNConv(2 * out_channels, out_channels,
                                       cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        #x = F.relu(self.conv3(x, edge_index))
        if args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        

kwargs = {'GAE': GAE, 'VGAE': VGAE}

model = kwargs[args.model](Encoder(args.input_dim, args.hidden1_dim)).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
print(model)

state = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}

# import scipy.sparse as sp
# from preprocessing import *
# pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
# norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# adj_train = adj
# adj_label = adj_train + sp.eye(adj_train.shape[0])
# adj_label = sparse_to_tuple(adj_label)
# adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
#                             torch.FloatTensor(adj_label[1]), 
#                             torch.Size(adj_label[2])).cuda()
# weight_mask = adj_label.to_dense().view(-1) == 1
# weight_tensor = torch.ones(weight_mask.size(0)).cuda()
# weight_tensor[weight_mask] = pos_weight



def model_load(model_load_path, load = False):
    if os.path.exists(model_load_path) and os.path.isfile(model_load_path):
        print("The pretrained model exists: {}".format(model_load_path))
        if load:
            print("Model Loading...")
            state = torch.load(model_load_path)
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            print("Done")
        else:
            #os.remove(model_load_path)
            print("The pretrained model is not loaded!")
    else:
        print("The pretrained model '{}' is not here.".format(model_load_path))

def model_save(state, model_save_path, save = False):
    if save:
        torch.save(state, model_save_path)
        print("The trained model is saved to {}".format(os.path.abspath(model_save_path)))
    else:
        print("The trained model has not saved.")

def train():
    model.train()
    loss_ave = 0
    for iter, data in enumerate(train_loader):
        # print(iter)
        # print(data)
        # data = data.to(dev)
        optimizer.zero_grad()
        z = model.encode(data.x.to(dev), edge_index_new)
        
        #A_pred = torch.sigmoid(torch.matmul(z,z.t())) #sig(Z*ZT)
        
        loss = model.recon_loss(z, edge_index_new)
        if args.model in ['VGAE']:
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
        
        # loss = log_lik = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        # if args.model == 'VGAE':
        #     kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)).sum(1).mean()
        #     loss -= kl_divergence
        loss_ave += loss
        loss.backward()
        optimizer.step()
    #print("{} iteration per epoch".format(iter))
    loss_ave /= iter
        
    return loss#, A_pred

def test(value, Adj_real, edge_index_new, threshold = 0.75):
    #data = data.to(dev)
    model.eval()
    with torch.no_grad():
        z = model.encode(value.x.to(dev), edge_index_new)
        A_pred = torch.sigmoid(torch.matmul(z,z.t())) #sig(Z*ZT)
        A_pred_result = A_pred.cpu().detach().numpy()
        A_pred_result[A_pred_result > threshold] = 1
        A_pred_result[A_pred_result < threshold] = 0
        #acc = np.sum(np.abs(Adj_real - A_pred_result)) / (Adj_real.shape[0] * Adj_real.shape[1])  
        #acc = np.count_nonzero((Adj_real - A_pred_result)==0) / (Adj_real.shape[0] * Adj_real.shape[1]) 
        accuracy = acc(A_pred_result, Adj_real)
    return A_pred_result, accuracy


def acc(A_pred_result, Adj_real):
    accuracy = (A_pred_result == Adj_real).sum() / (Adj_real.shape[0] * Adj_real.shape[1])
    return accuracy


if args.remove_model_folder:
    try:
        shutil.rmtree(args.model_folder)
        print("The models folder is deleted.")
        os.mkdir(args.model_folder)
    except OSError as e:
        print("Error: %s : %s" % (args.model_folder, e.strerror))
        os.mkdir(args.model_folder)
else:
    if not os.path.isdir(args.model_folder):
        os.mkdir(args.model_folder)
        print("The models folder is not here.")

# read the pretrain model
model_load(args.model_load_path, args.load)

if args.is_train:
    print("Training Time:") 
    for epoch in range(args.num_epoch):
        loss = train()
        print("Epoch {} ===================> loss: {}".format(epoch, loss))
        
        #save models
        if epoch % args.save_per_epoch == 0:
        #if loss <= 1.53:
            save_path = args.model_folder + 'model_epoch_' + str(epoch) + '.pth'
            model_save(state, save_path, args.save)
    print("Training Done.")
else:
    print("Training Skip")
      

print("Testing Time.\nHope you will enjoy!") 
for key, value in test_data_images.items():
    A_pred_result, accuracy = test(value, rd.adj2data(adj)[2], edge_index_new, threshold = args.threshold)
    print("The test accuracy of '{}' is {}.".format(key, accuracy))
    con = rd.adj2connection(sp.csr_matrix(A_pred_result))
    temp_coor = all_coor_dict[key]
    rd.show_connection(args.img_name, temp_coor, con)
    
    
