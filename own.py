import readim as rd
import numpy as np
#import sys
from sklearn.preprocessing import normalize
from torch_geometric.data import Data
import os
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
import args
import model
import scipy.sparse as sp
from preprocessing import sparse_to_tuple
import shutil
import reorder_adj as ra


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# reorder_all_coor_dict, reorder_adj_all_dict = ra.reorder_all(all_coor_dict, args.n_point, ori_cconnection_file_path = args.file_path, show = False)

feat_shape = (args.n_point, args.num_node_features)

all_coor_dict_train_ori = rd.load_all_coor(args.coor_file_path_train)
all_coor_dict_test_ori = rd.load_all_coor(args.coor_file_path_test)

adj_ori, graph_ori = rd.load_data_txt(args.file_path)
adj_all_dict_train_ori = {}
for key, value in all_coor_dict_train_ori.items():
    adj_all_dict_train_ori[key] = adj_ori


reorder_all_coor_dict_train, reorder_adj_all_dict_train = ra.reorder_all(all_coor_dict_train_ori, args.n_point, ori_cconnection_file_path = args.file_path, show = False)
all_coor_dict_test, reorder_adj_all_dict_test = ra.reorder_all(all_coor_dict_test_ori, args.n_point, ori_cconnection_file_path = args.file_path, show = False)


#num_train = len(all_coor_dict) - int(len(all_coor_dict) * args.test_split)
# if num_train % args.batch_size != 0:
#     num_train = int(num_train / args.batch_size) * args.batch_size - 1
    

# adj, graph = rd.load_data_txt(args.file_path)
# edge_index = torch.tensor(rd.adj2connection(adj), dtype=torch.long)



label = [0]



def get_data(all_coor_dict, feat_shape, adj_all_dict):
    '''
    ----------
    Parameters
    ----------
    all_coor_dict : dict
        {"1.jpg":[coordinates], "2.jpg":[coordinates], ...}
    feat_shape : tuple
        feat_shape = (args.n_point, args.num_node_features).
    adj_all_dict : dict
        {"1.jpg":[reorder_adj(csr_matrix)], "2.jpg":[coordinates], ...}
    ----------
    Returns
    ----------
    data_list : list
        [Data(adj=[10, 10], adj_label=[10, 10], edge_index=[2, 21], img_name=00001241.jpg, norm=0.6329113924050633, weight_tensor=[100], x=[10, 2], y=[1]),
         Data(adj=[10, 10], ...) 
         ...]
    data_images : dict
        {"1.jpg": Data(adj=[10, 10], adj_label=[10, 10], edge_index=[2, 21], img_name=00001241.jpg, norm=0.6329113924050633, weight_tensor=[100], x=[10, 2], y=[1]),
         "2.jpg": ..., 
         ...]

    '''
    data_list = []
    data_images = {}
    for key, value in all_coor_dict.items():

        features = rd.get_features(feat_shape, key, all_coor_dict)
        features = sparse_to_tuple(features.tocoo())
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                            torch.FloatTensor(features[1]), 
                            torch.Size(features[2])).to(dev)
        
        edge_index = torch.tensor(rd.adj2connection(adj_all_dict[key]), dtype=torch.long)
        
        adj = adj_all_dict[key]
        adj_norm, adj_label, norm, weight_tensor = rd.data_process(adj)
        
        data = Data(x = features, edge_index=edge_index.t().contiguous(), norm = norm, y = label, adj = adj_norm, img_name = key, weight_tensor = weight_tensor, adj_label = adj_label)
        data_list.append(data)
        data_images[key] = data
    return data_list, data_images

reorder_train_data_list, reorder_train_data_images = get_data(reorder_all_coor_dict_train, feat_shape, reorder_adj_all_dict_train)
train_data_list_ori, ori_train_data_images_ori = get_data(all_coor_dict_train_ori, feat_shape, adj_all_dict_train_ori)
test_data_list, test_data_images = get_data(all_coor_dict_test, feat_shape, reorder_adj_all_dict_test)

       
# train_loader = DataLoader(reorder_train_data_list, batch_size = args.batch_size, shuffle=True)
# test_loader = DataLoader(test_data_list, shuffle=False)
print("Keys in data are {}.".format(train_data_list_ori[0].keys))
print("Num. of nodes is {}.".format(train_data_list_ori[0].num_nodes))
print("Num. of edges is {}.".format(train_data_list_ori[0].num_edges))
print("Num. of node features is {}.".format(train_data_list_ori[0].num_node_features))


# init model and optimizer
#model = getattr(model,args.model)(adj_norm).to(dev)

model = model.GAE().to(dev)
optimizer = Adam(model.parameters(), lr=args.learning_rate)

print("The model we will use is: " + str(args.model) + '.\nThe architecture: ')
print(model)

state = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}

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


# train model
def train(model, ori_train_data_images_ori, num_epoch, save_per_epoch, feat_shape, all_coor_dict_train_ori, n_point, ori_cconnection_file_path = args.file_path, save = True):
    for epoch in range(num_epoch):
        t = time.time()
        loss_ave = 0.0
        #for data in reorder_train_data_list:
        reorder_all_coor_dict_train, reorder_adj_all_dict_train = ra.reorder_all(all_coor_dict_train_ori, args.n_point, ori_cconnection_file_path = args.file_path, show = False)
        _, reorder_train_data_images = get_data(reorder_all_coor_dict_train, feat_shape, reorder_adj_all_dict_train)
        for key, value in ori_train_data_images_ori.items():
            
            A_pred = model(ori_train_data_images_ori[key].x, reorder_train_data_images[key].x, ori_train_data_images_ori[key].adj, reorder_train_data_images[key].adj).to(dev)
            # print(data.adj)
            optimizer.zero_grad()

            #print(reorder_train_data_images[key].adj)
            loss  = ori_train_data_images_ori[key].norm * F.binary_cross_entropy(A_pred.view(-1), ori_train_data_images_ori[key].adj_label.to_dense().view(-1), weight = ori_train_data_images_ori[key].weight_tensor)

            if args.model == 'VGAE':
                kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)).sum(1).mean()
                loss -= kl_divergence
        
            loss.backward()
            optimizer.step()

            train_acc = rd.get_acc(A_pred,ori_train_data_images_ori[key].adj_label)

            loss_ave += loss.item()

        # print(data.adj_label)
        # print(A_pred.view(-1))
    
        
        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
        #       "train_acc=", "{:.5f}".format(train_acc), "time=", "{:.5f}".format(time.time() - t))
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_ave / len(reorder_train_data_list)),
              "train_acc=", "{:.5f}".format(train_acc), "time=", "{:.5f}".format(time.time() - t))
            
        #save models
        if save == True:
            if epoch % save_per_epoch == 0:
            #if loss <= 1.53:
                save_path = args.model_folder + 'model_epoch_' + str(epoch) + '.pth'
                model_save(state, save_path, args.save)
 

def test(adj_ori, value, adj_label, threshold = 0.75):
    #data = data.to(dev)
    model.eval()
    with torch.no_grad():
        z_real = model.encode_ori(value.x, adj_ori)
        z_reorder = model.encode_reo(value.x, value.adj)
        z = args.alpha * z_real + args.delta * z_reorder
        A_pred = torch.sigmoid(torch.matmul(z,z.t())) #sig(Z*ZT)
        # A_pred_result = A_pred.cpu().detach().numpy()
        # A_pred_result[A_pred_result > threshold] = 1
        # A_pred_result[A_pred_result < threshold] = 0

        test_acc = rd.get_acc(A_pred,adj_label)
    return A_pred, test_acc.item()

def show(A_pred, image, all_coor_dict):
    threshold = 0.76
    result = A_pred.cpu().detach().numpy()
    result[result > threshold] = 1
    result[result < threshold] = 0
    
    con = rd.adj2connection(sp.csr_matrix(result))
    temp_coor = all_coor_dict[image]
    rd.show_connection(image, temp_coor, con)
    
# read the pretrain model
model_load(args.model_load_path, args.load)

# print("Choosing the unsatisfied images whose accuracy is less than 0.85.")
# temp_data_images = {}
# temp_data_list = []
# for key, value in reorder_train_data_images.items():
#     A_pred_result, accuracy = test(value, adj_label, threshold = args.threshold)
#     if accuracy <= 0.85:
#         temp_data_images[key] = value
#         temp_data_list.append(value)
# print("Done.")
# print("There exits {} images will be training.".format(len(temp_data_list)))


if args.is_train == True:
    train(model, ori_train_data_images_ori, args.num_epoch, args.save_per_epoch, feat_shape, all_coor_dict_train_ori, args.n_point, ori_cconnection_file_path = args.file_path, save = True)  

print("Testing Time.\nHope you will enjoy!") 
for key, value in test_data_images.items():
    #print(value.adj)
    A_pred_result, accuracy = test(adj_ori, value, value.adj_label, threshold = args.threshold)
    #A_pred_result = test(value, threshold = args.threshold)
    print("The test accuracy of '{}' is {}.".format(key, accuracy))
    show(A_pred_result, key, all_coor_dict_test)     

# print("Test on reorder train.") 
# for key, value in reorder_train_data_images.items():
#     #print(value.adj)
#     A_pred_result, accuracy = test(adj_ori, value, value.adj_label, threshold = args.threshold)
#     #A_pred_result = test(value, threshold = args.threshold)
#     print("The test accuracy of '{}' is {}.".format(key, accuracy))
#     show(A_pred_result, key, all_coor_dict_test)     