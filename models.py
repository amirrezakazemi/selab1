import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.autograd import Variable
# GINConv model for regression
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()
        self.grads = {}
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers for drug
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        # linear layer for drug
        self.fc1_xd = Linear(dim, output_dim)

        #embedding layer for protein
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.embedding_xt.weight.requires_grad = True
        # 1D convolution for protein
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        # linear layer for protein
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        u = F.relu(self.conv1(x, edge_index))
        u = self.bn1(u)
        u = F.relu(self.conv2(u, edge_index))
        u = self.bn2(u)
        u = F.relu(self.conv3(u, edge_index))
        u = self.bn3(u)
        u = F.relu(self.conv4(u, edge_index))
        u = self.bn4(u)
        u = F.relu(self.conv5(u, edge_index))
        u = self.bn5(u)
        u = global_add_pool(u, batch)
        u = F.relu(self.fc1_xd(u))
        u = F.dropout(u, p=0.2, training=self.training)

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        xc = torch.cat((u, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

    def save_grad(self, name):
      def hook(grad):
        self.grads[name] = grad
      return hook



# GINConv model for classification

class GINConvNetClassification(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNetClassification, self).__init__()
        dim = 32
        self.n_output = n_output
        # variables to store gradient
        self.grads = {}
        self.grad_x = 0
        self.grad_embedded_xt = 0
        self.grad_conv_xt = 0
        self.grad_conv = 0
        self.conv = 0
        self.conv_xt = 0

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # convolution layers for drug
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        #linear layer for drug
        self.fc1_xd = Linear(dim, output_dim)

        # embed layer for protein
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.embedding_xt.weight.requires_grad = False
        # 1D convolution on protein sequence
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        # linear layer for protein
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.classifyout = nn.Linear(256, 2 * self.n_output)        # n_output = 2 for classification task

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # this helps to handle 1 input
        try:
          batch = data.batch 
        except:
          batch = torch.Tensor([0]).long().to('cuda:0')

        target = data.target
        self.grad_x = x


        u = F.relu(self.conv1(x, edge_index))
        u = self.bn1(u)
        u = F.relu(self.conv2(u, edge_index))
        u = self.bn2(u)
        u = F.relu(self.conv3(u, edge_index))
        u = self.bn3(u)
        u = F.relu(self.conv4(u, edge_index))
        u = self.bn4(u)
        u = F.relu(self.conv5(u, edge_index))
        u = self.bn5(u)
        self.grad_conv = u
        self.conv = u

        u = global_add_pool(u, batch)
        u = F.relu(self.fc1_xd(u))
        u = F.dropout(u, p=0.2, training=self.training)

        embedded_xt = self.embedding_xt(target)
        embedded_xt.requires_grad = True
        self.grad_embedded_xt = embedded_xt
        
        conv_xt = self.conv_xt_1(embedded_xt) 
        self.conv_xt = conv_xt 
        self.grad_conv_xt = conv_xt

        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        xc = torch.cat((u, xt), 1)

        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.classifyout(xc)
        return out

    def save_grad(self, name):
      def hook(grad):
        self.grads[name] = grad
      return hook








