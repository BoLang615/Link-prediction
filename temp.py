import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

'''
data.x: Node feature matrix with shape [num_nodes, num_node_features]

data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long

data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]

data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]

data.pos: Node position matrix with shape [num_nodes, num_dimensions]
'''


x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index.t().contiguous())

#edge_index = torch.tensor([[0, 1, 1, 2],[1, 0, 2, 1]], dtype=torch.long)
#data = Data(x=x, edge_index=edge_index)

print(data.keys)

for key, item in data:
    print("{} found in data".format(key))

dataset = Planetoid(root='/tmp/Cora', name='Cora')
len(dataset)
print(dataset.num_node_features)
print(dataset.num_classes)
print(dataset[0].keys)
print(dataset[0]['x'].shape)

#adj, features, graph = load_data(dataset = 'cora')

label = dataset[0].y.numpy()


# from dgl.data import MiniGCDataset
# import matplotlib.pyplot as plt
# import networkx as nx
# # A dataset with 80 samples, each graph is
# # of size [10, 20]
# dataset = MiniGCDataset(80, 10, 20)
# graph, label = dataset[0]
# fig, ax = plt.subplots()
# nx.draw(graph.to_networkx(), ax=ax)
# ax.set_title('Class: {:d}'.format(label))
# plt.show()