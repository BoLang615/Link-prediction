### CONFIGS ###
#dataset = 'cora'
model = 'GAE'

input_dim = 2 #1433
hidden1_dim = 256#32
hidden2_dim = 128#16
use_feature = True

num_epoch = 200000
alpha = 1
delta = 1

#learning_rate = 0.000001
learning_rate = 0.0001
batch_size = 1

file_path = 'connection.txt'
#coor_file_path = 'chairs_coor_test.txt'
coor_file_path_train = 'chairs_coor_test.txt'
coor_file_path_test = 'chairs_coor_test.txt'
model_folder = 'models/'

# model_load_path = 'model_epoch_68000.pth'
# model_load_path = 'model_epoch_27320.pth'
model_load_path = 'model_epoch_47280.pth'



test_split = 0.1
remove_model_folder = True
load = True
save = True
is_train = False

n_point = 10
num_node_features = 2
threshold = 0.75
save_per_epoch = 20


