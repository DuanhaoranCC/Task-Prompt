# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Email   :
# @File    : model.py
# @Software: PyCharm
# @Note    :
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import global_mean_pool, global_add_pool, TemporalEncoding, GINConv, GCNConv, BatchNorm, \
    global_max_pool, LayerNorm, SAGEConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, dropout_edge, mask_feature, \
    to_networkx, degree
from torch_geometric.data import Data, Batch
import numpy as np
import networkx as nx
import copy
from load_data import text_to_vector
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.font_manager as fm


class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(TDrumorGCN, self).__init__()

        self.conv1 = GCNConv(in_feats, 128)
        self.conv2 = GCNConv(128, out_feats)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)

        h = self.conv2(h, edge_index)

        hs = global_add_pool(h, batch)

        return hs, h


class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(BUrumorGCN, self).__init__()

        self.conv1 = GCNConv(in_feats, 128)
        self.conv2 = GCNConv(128, out_feats)

    def forward(self, x, edge_index, batch):
        edge_index = torch.flip(edge_index, dims=[0])

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)

        h = self.conv2(h, edge_index)

        hs = global_add_pool(h, batch)

        return hs, h


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BiGCN_graphcl(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(BiGCN_graphcl, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, out_feats)

        prompt_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(out_feats * 2 * 2 + prompt_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.prompt = nn.Parameter(torch.FloatTensor(1, prompt_dim))
        nn.init.xavier_uniform_(self.prompt)
        self.p_task = nn.Parameter(torch.FloatTensor(1, prompt_dim))
        nn.init.xavier_uniform_(self.p_task)
        # self.p_task1 = nn.Parameter(torch.FloatTensor(1, in_feats))
        # nn.init.xavier_uniform_(self.p_task1)

    def gnn_backbone(self, x, edge_index, batch):

        x = x
        td_hs, td_h = self.TDrumorGCN(x, edge_index, batch)
        bu_hs, bu_h = self.BUrumorGCN(x, edge_index, batch)
        h = torch.cat([td_hs, bu_hs], dim=1)
        return h

    def forward(self, data_list):
        """
        【预训练】的前向传播函数。
        它的唯一职责是：为输入的每个图，计算出其图级别的表示。
        真正的配对和损失计算，将在 loss_graphcl 中完成，以匹配您的 pre_train 函数。
        """
        # all_graph_embs = []
        # for data in data_list:
        #     graph_embs = self.gnn_backbone(data.x, data.edge_index, data.batch)
        #     all_graph_embs.append(graph_embs)
        #
        # return torch.cat(all_graph_embs, dim=0)

        graph_embs = self.gnn_backbone(data_list.x, data_list.edge_index, data_list.batch)
        return graph_embs

    # def loss_graphcl(self, h1, h2):
    #     """
    #     【预训练】的损失计算函数。
    #     严格遵循您的“拼接池化表示+提示”架构。
    #     h1: 来自第一种增强的所有图的表示。
    #     h2: 来自第二种增强的所有图的表示。
    #     """
    #     batch_size = h1.size(0)
    #     device = h1.device
    #
    #     # --- 1. 构建正样本对 (标签=0) ---
    #     # h1[i] 和 h2[i] 来自同一个原始图，是正样本
    #     pos_input = torch.cat([h1, h2, self.prompt.expand(batch_size, -1)], dim=1)
    #     pos_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    #
    #     # --- 2. 构建负样本对 (标签=1) ---
    #     # 通过打乱 h2 来创建负样本对：h1[i] 和 h2[j] (i != j)
    #     h2_shuffled = h2[torch.randperm(batch_size)]
    #     neg_input = torch.cat([h1, h2_shuffled, self.prompt.expand(batch_size, -1)], dim=1)
    #     neg_labels = torch.ones(batch_size, dtype=torch.long, device=device)
    #
    #     # --- 3. 组合、预测并计算损失 ---
    #     total_input = torch.cat([pos_input, neg_input], dim=0)
    #     total_labels = torch.cat([pos_labels, neg_labels], dim=0)
    #
    #     # 喂给 MLP 进行二分类预测
    #     logits = self.mlp(total_input)
    #
    #     # 计算交叉熵损失
    #     loss = F.cross_entropy(logits, total_labels)
    #
    #     return loss

    def loss_graphcl(self, h1, h2):
        """
        【全新重写】的损失计算函数，实现了高效的 1-vs-(N-1) 负采样。
        它完美地适配您“拼接池化表示+提示 -> MLP分类”的核心架构。

        h1: 来自第一种增强的所有图的表示，形状为 [B, D]。
        h2: 来自第二种增强的所有图的表示，形状也为 [B, D]。
        """
        batch_size = h1.size(0)
        device = h1.device

        # --- 1. 构建一个 [B, B] 的“比较矩阵” ---
        # 我们的目标是，让每个 h1[i] 都与所有的 h2[j] 进行一次比较。

        # 扩展 h1: 将每个 h1[i] 复制 B 次，形成 B 组，每组都是 h1[i]。
        # [B, D] -> [B, 1, D] -> [B, B, D]
        h1_expanded = h1.unsqueeze(1).expand(-1, batch_size, -1)

        # 扩展 h2: 将整个 h2 复制 B 次。
        # [B, D] -> [1, B, D] -> [B, B, D]
        h2_expanded = h2.unsqueeze(0).expand(batch_size, -1, -1)

        # 现在，h1_expanded[i, j] 是 h1[i]，而 h2_expanded[i, j] 是 h2[j]。
        # 我们已经为 B*B 次比较准备好了输入。

        # --- 2. 准备拼接，并喂给 MLP ---
        # 将 B*B 对图表示拉平成一个长向量
        # [B, B, D] -> [B*B, D]
        h1_flat = h1_expanded.reshape(batch_size * batch_size, -1)
        h2_flat = h2_expanded.reshape(batch_size * batch_size, -1)

        # 扩展 prompt 以匹配 B*B 的大小
        prompt_batch = self.prompt.expand(batch_size * batch_size, -1)

        # 拼接所有输入
        combined_input = torch.cat([h1_flat, h2_flat, prompt_batch], dim=1)

        # 一次性通过 MLP 得到所有 B*B 对比较的 logits
        # logits 的形状为 [B*B, 2]
        logits = self.mlp(combined_input)

        # --- 3. 创建对应的标签 ---
        # 在这个 B*B 的比较矩阵中，只有对角线上的元素 (h1[i], h2[i]) 是正样本对。
        # 我们的标签需要反映这一点。
        # 假设标签 0 代表“相似”（正样本），标签 1 代表“不相似”（负样本）。

        # 创建一个全为 1 (不相似) 的标签张量
        labels = torch.ones(batch_size * batch_size, dtype=torch.long, device=device)

        # 找到对角线元素的索引，并将它们的标签设置为 0 (相似)
        # 对角线索引是 [0, B+1, 2B+2, ..., B*B-1]
        diag_indices = torch.arange(batch_size, device=device) * (batch_size + 1)
        labels[diag_indices] = 0

        # --- 4. 计算最终的交叉熵损失 ---
        loss = F.cross_entropy(logits, labels)

        return loss

    def freeze_backbone(self):
        for m in [self.TDrumorGCN, self.BUrumorGCN]:
            for p in m.parameters():
                p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False

    def finetune(self, batch):
        """
        【全新重写】的微调函数，以适应只有 batch 作为输入的场景。
        """
        num_graphs = batch.num_graphs
        x = batch.x
        td_hs, td_h = self.TDrumorGCN(x, batch.edge_index, batch.batch)
        bu_hs, bu_h = self.BUrumorGCN(x, batch.edge_index, batch.batch)
        h = torch.cat([td_h, bu_h], dim=1)

        h_structures = global_add_pool(h, batch.batch)
        # ======================================================================
        # 阶段 3: 在批次内部构建原型
        # ======================================================================

        # 3a. 根据真实标签 batch.y 分离结构表示
        # 假设 1 代表谣言, 0 代表非谣言
        mask_rumor = (batch.y == 1)
        mask_non_rumor = (batch.y == 0)

        h_rumor_structures = h_structures[mask_rumor]
        h_non_rumor_structures = h_structures[mask_non_rumor]

        # 3b. 计算原型 (对当前批次内的同类样本求平均)
        # 添加一个微小的数以防止在某个批次中没有某一类样本时出现除以零的错误
        proto_rumor = h_rumor_structures.mean(dim=0)
        proto_non_rumor = h_non_rumor_structures.mean(dim=0)

        # 处理边界情况：如果批次中只有一个类别，则用自身的表示作为另一类的原型
        if h_rumor_structures.size(0) == 0:
            proto_rumor = torch.zeros_like(proto_non_rumor)  # 或者使用一个更合理的默认值
        if h_non_rumor_structures.size(0) == 0:
            proto_non_rumor = torch.zeros_like(proto_rumor)

        # ======================================================================
        # 阶段 4: 对批次内所有图进行“有参照的结构对比”
        # ======================================================================

        # 4a. 将查询图的结构与“谣言原型”进行对比
        proto_rumor_batch = proto_rumor.expand(num_graphs, -1)
        comparison_input_rumor = torch.cat([
            h_structures,
            proto_rumor_batch,
            self.p_task.expand(num_graphs, -1)
        ], dim=1)
        s_rumor = self.mlp(comparison_input_rumor)

        # 4b. 将查询图的结构与“非谣言原型”进行对比
        proto_non_rumor_batch = proto_non_rumor.expand(num_graphs, -1)
        comparison_input_non_rumor = torch.cat([
            h_structures,
            proto_non_rumor_batch,
            self.p_task.expand(num_graphs, -1)
        ], dim=1)
        s_non_rumor = self.mlp(comparison_input_non_rumor)

        # 4c. 组合最终的 Logits
        score_rumor_similarity = s_rumor[:, 0]
        score_non_rumor_similarity = s_non_rumor[:, 0]

        logits = torch.stack([score_non_rumor_similarity, score_rumor_similarity], dim=1)

        return logits


class BiGCN_individual(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(BiGCN_individual, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, out_feats)
        self.proj_head = nn.Sequential(nn.Linear(out_feats * 2, 256), nn.ReLU(inplace=True),
                                       nn.Linear(256, 128))

        self.time_encoder = TemporalEncoding(100)
        self.fc = nn.Linear(out_feats * 2, 2)

    def forward(self, data):
        # edge_attr = np.log(1 + np.abs(data.edge_attr))
        # edge_attr = self.time_encoder(edge_attr)

        x = data.x

        TD_x1, _ = self.TDrumorGCN(x, data.edge_index, data.batch)
        BU_x1, _ = self.BUrumorGCN(x, data.edge_index, data.batch)
        h = torch.cat((BU_x1, TD_x1), 1)
        # h = BU_x1

        h = self.proj_head(h)
        return h

    def loss_graphcl(self, x1, x2, mean=True):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss

    def bigcn(self, data):
        TD_x, _ = self.TDrumorGCN(data.x, data.edge_index, data.batch)
        BU_x, _ = self.BUrumorGCN(data.x, data.edge_index, data.batch)
        x = torch.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

    def get_embeds(self, data):
        TD_x1, _ = self.TDrumorGCN(data.x, data.edge_index, data.batch)
        BU_x1, _ = self.BUrumorGCN(data.x, data.edge_index, data.batch)
        h = torch.cat((BU_x1, TD_x1), 1)
        # h = BU_x1

        return h


class UPFD_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_class, concat=True):
        super().__init__()
        self.concat = concat

        self.conv1 = GCNConv(in_channels, out_channels)
        # self.conv1 = SAGEConv(in_channels, hidden_channels)
        # self.conv1 = GATConv(in_channels, hidden_channels)

        if self.concat:
            self.lin0 = Linear(in_channels, out_channels)
            self.lin1 = Linear(2 * out_channels, out_channels)

        self.lin2 = Linear(out_channels, num_class)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index = to_undirected(edge_index)
        h = self.conv1(x, edge_index).relu()
        h = global_max_pool(h, batch)

        if self.concat:
            # Get the root node (tweet) features of each graph:
            root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            root = torch.cat([root.new_zeros(1), root + 1], dim=0)
            news = x[root]

            news = self.lin0(news).relu()
            h = self.lin1(torch.cat([news, h], dim=-1)).relu()

        h = self.lin2(h)
        return h.log_softmax(dim=-1)


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss = - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def mask(x, mask_rate=0.5):
    """
    Function to mask a subset of nodes in a graph.

    Args:
        x (torch.Tensor): Node feature matrix.
        mask_rate (float): The rate of nodes to be masked.

    Returns:
        torch.Tensor: Indices of the masked nodes.
    """
    num_nodes = x.size(0)  # Number of nodes in the graph
    perm = torch.randperm(num_nodes, device=x.device)  # Random permutation of node indices
    num_mask_nodes = int(mask_rate * num_nodes)  # Number of nodes to mask
    mask_nodes = perm[:num_mask_nodes]  # Select the indices of masked nodes

    return mask_nodes


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, num_layers=2):
        super(Encoder, self).__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.act = nn.ModuleList()

        # Initialize GIN layers with MLPs
        for layer in range(num_layers):
            if layer == 0:
                mlp = MLP(in_dim, hidden, out_dim)
            else:
                mlp = MLP(out_dim, hidden, out_dim)

            self.ginlayers.append(GINConv(mlp))
            self.batch_norms.append(BatchNorm(out_dim))
            self.act.append(nn.PReLU())

    def forward(self, x, edge_index, batch):
        output = []

        for i, layer in enumerate(self.ginlayers):
            x = layer(x, edge_index)  # Message passing
            x = self.batch_norms[i](x)  # Batch normalization
            x = F.relu(x)  # Activation function
            pooled = global_add_pool(x, batch)  # Global pooling (sum pooling)
            output.append(pooled)

        return x, torch.cat(output, dim=1)


class GAMC(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()

        self.encoder = Encoder(in_dim, out_dim, hidden, 2)
        self.decoder = Encoder(out_dim, in_dim, hidden, 1)
        self.criterion = self.setup_loss_fn("sce")

        self.fc = nn.Linear(out_dim*2, 2)

    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=1)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, data):
        loss = 0
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index = to_undirected(edge_index)
        edge_index, _ = dropout_edge(edge_index, p=0.2)
        mask_nodes = mask(x, mask_rate=0.5)

        x1 = x.clone()
        x1[mask_nodes] = 0.0
        h, gh = self.encoder(x1, edge_index, batch)

        re_h = h.clone()
        re_h[mask_nodes] = 0.0
        re_x1, _ = self.decoder(re_h, edge_index, batch)
        loss1 = self.criterion(re_x1[mask_nodes], x[mask_nodes].detach())

        ################################################################################
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index = to_undirected(edge_index)
        edge_index, _ = dropout_edge(edge_index, p=0.2)
        mask_nodes = mask(x, mask_rate=0.5)

        x1 = x.clone()
        x1[mask_nodes] = 0.0
        h, gh = self.encoder(x1, edge_index, batch)

        re_h = h.clone()
        re_h[mask_nodes] = 0.0
        re_x2, _ = self.decoder(re_h, edge_index, batch)
        loss2 = self.criterion(re_x2[mask_nodes], x[mask_nodes].detach())
        ############################################################################
        # Contrastive
        cl_loss = self.criterion(re_x2, re_x1)
        loss += loss1 + loss2 + cl_loss * 0.1

        return loss

    def get_embeds(self, data):
        h, gh = self.encoder(data.x, data.edge_index, data.batch)

        return gh

    def finetune(self, data):
        h, gh = self.encoder(data.x, data.edge_index, data.batch)
        x = self.fc(gh)
        return F.log_softmax(x, dim=-1)


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))


class Encoder1(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(Encoder1, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, out_feats)

    def forward(self, data, x):

        TD_x1, h1 = self.TDrumorGCN(x, data.edge_index, data.batch)
        BU_x1, h2 = self.BUrumorGCN(x, data.edge_index, data.batch)
        hs = torch.cat((BU_x1, TD_x1), 1)
        h = torch.cat((h1, h2), 1)

        return h, hs

    def loss_graphcl(self, x1, x2, mean=True):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss


def mask1(x, batch, mask_rate=0.5):
    """
    Mask a subset of nodes in a batched graph, ensuring the 0th node in each graph is not masked.

    Args:
        x (torch.Tensor): Node feature matrix.
        batch (torch.Tensor): Batch vector indicating the graph each node belongs to.
        mask_rate (float): The rate of nodes to be masked.

    Returns:
        torch.Tensor: Indices of the masked nodes.
    """
    mask_nodes = []  # Store indices of masked nodes
    num_graphs = batch.max().item() + 1  # Number of graphs in the batch

    for graph_idx in range(num_graphs):
        # Get the nodes belonging to the current graph
        graph_nodes = (batch == graph_idx).nonzero(as_tuple=True)[0]
        # Exclude the 0th node of the graph
        non_zero_nodes = graph_nodes[1:] if len(graph_nodes) > 1 else graph_nodes
        # Compute the number of nodes to mask
        num_mask_nodes = int(mask_rate * len(non_zero_nodes))
        # Randomly select nodes to mask
        perm = torch.randperm(len(non_zero_nodes), device=x.device)
        masked_nodes = non_zero_nodes[perm[:num_mask_nodes]]
        mask_nodes.append(masked_nodes)

    # Concatenate masked node indices from all graphs
    return torch.cat(mask_nodes)


class CFOP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rate, alpha, hidden=64):
        super().__init__()

        # self.online_encoder = Encoder(in_dim, out_dim, hidden, 2)
        # self.target_encoder = Encoder(in_dim, out_dim, hidden, 2)
        self.online_encoder = Encoder1(in_dim, out_dim)
        self.target_encoder = Encoder1(in_dim, out_dim)
        self.criterion = self.setup_loss_fn("sce")
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

        self.rate = rate
        self.alpha = alpha
        self.fc = nn.Linear(out_dim*2, 2)
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=1)
        else:
            raise NotImplementedError
        return criterion

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, data):
        loss = 0
        x, edge_index, batch = data.x, data.edge_index, data.batch
        mask_nodes = mask1(x, batch, mask_rate=self.rate)
        x1 = x.clone()
        x1[mask_nodes] = 0.0
        x1[mask_nodes] += self.enc_mask_token

        h1, gh1 = self.online_encoder(data, x1)
        with torch.no_grad():
            h2, gh2 = self.target_encoder(data, x)

        loss += self.criterion(h1[mask_nodes], h2[mask_nodes].detach()) + \
                self.criterion(gh1, gh2.detach())
        ##################################################################################
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # # edge_index = to_undirected(edge_index)
        # mask_nodes = mask1(x, batch, mask_rate=self.rate)
        # x1 = x.clone()
        # x1[mask_nodes] = 0.0
        # x1[mask_nodes] += self.enc_mask_token
        #
        # h1, gh1 = self.online_encoder(data, x1)
        # with torch.no_grad():
        #     h2, gh2 = self.target_encoder(data, x)
        #
        # loss = self.criterion(h1[mask_nodes], h2[mask_nodes].detach()) + \
        #        self.criterion(gh1, gh2.detach())

        return loss

    def get_embeds(self, data):
        h, gh = self.online_encoder(data, data.x)

        return gh

    def finetune(self, data):
        h, gh = self.online_encoder(data.x, data.edge_index, data.batch)
        x = self.fc(gh)
        return F.log_softmax(x, dim=-1)


class GCNEncoder(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout=0.5):
        """
        标准两层 GCN 作为编码器

        参数:
        - in_feats: 输入特征维度
        - hidden_dim: 隐藏层维度
        - out_feats: 输出特征维度
        - dropout: Dropout 比例
        """
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)  # 第一层 GCN
        self.conv2 = GCNConv(hidden_dim, out_feats)  # 第二层 GCN
        self.dropout = nn.Dropout(dropout)  # Dropout 以减少过拟合

    def forward(self, x, edge_index):
        """
        前向传播

        输入:
        - x: 节点特征矩阵，形状为 [num_nodes, in_feats]
        - edge_index: 边索引，形状为 [2, num_edges]

        输出:
        - 节点的最终嵌入，形状为 [num_nodes, out_feats]
        """
        x = self.conv1(x, edge_index)  # 第一层 GCN
        x = F.relu(x)  # ReLU 激活函数
        x = self.dropout(x)  # Dropout 处理
        x = self.conv2(x, edge_index)  # 第二层 GCN
        return x


class OFA(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(OFA, self).__init__()

        self.gcn = GCNEncoder(in_feats, 128, out_feats)

        # 文本编码器（用于 Prompt Nodes 和 Class Nodes）
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', cache_dir="./")
        self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base', cache_dir="./")

        # 定义类别节点 (Class Nodes)
        class_descriptions = [
            "Prompt node. Rumor classification label. Real News",
            "Prompt node. Rumor classification label. Rumor or Fake News"
        ]
        self.class_nodes = [self.text_to_vector(desc) for desc in class_descriptions]

        # 定义任务提示节点 (Prompt Node)
        self.prompt_node = self.text_to_vector(
            "Prompt node. Task description. Detect whether a given post is spreading misinformation or real news "
            "based on its content and social interactions."
        )

        # 分类层（用于最终的图分类）
        self.fc = nn.Linear(out_feats, 2)

    def text_to_vector(self, text):
        """ 使用 XLM-Roberta 将文本转换为向量 """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().squeeze(0)  # 取 CLS token 表示

    def construct_prompted_graph(self, data):
        """
        Constructs the prompted graph by:
            1. Adding new Prompt Nodes for each graph in the batch.
            2. Adding new Class Nodes for each graph in the batch.
            3. Connecting all original graph nodes to their corresponding Prompt Node.
            4. Connecting the Prompt Node to its corresponding Class Nodes.
        """
        num_graphs = data.num_graphs  # Number of graphs in batch
        device = data.x.device

        # Get original batch node structure
        num_nodes_per_graph = torch.bincount(data.batch)  # Nodes per graph
        cumulative_nodes = torch.cat([torch.tensor([0], device=device), num_nodes_per_graph.cumsum(dim=0)])
        total_original_nodes = cumulative_nodes[-1].item()  # Total original nodes in batch

        prompt_nodes = torch.tensor(self.prompt_node).unsqueeze(0).repeat(num_graphs, 1).clone().to(
            device)  # [num_graphs, d]
        class_nodes = torch.stack(self.class_nodes).repeat(num_graphs, 1, 1).clone().view(-1,
                                                                                          torch.stack(
                                                                                              self.class_nodes).size(
                                                                                              -1)).to(device)

        all_x, all_edges = [], []
        prompt_indices, class_indices = [], []

        for i in range(num_graphs):
            start_idx = cumulative_nodes[i].item()
            end_idx = cumulative_nodes[i + 1].item()

            # New node indices (AFTER the original nodes)
            prompt_idx = total_original_nodes + i  # New prompt node index
            class_start_idx = total_original_nodes + num_graphs + (i * len(self.class_nodes))  # First class node index

            # Connect all nodes to Prompt Node
            graph_edges = torch.stack([
                torch.arange(start_idx, end_idx, device=device),
                torch.full((end_idx - start_idx,), prompt_idx, device=device)
            ], dim=1).T

            # Connect Prompt Node to Class Nodes
            class_edges = torch.stack([
                torch.full((len(self.class_nodes),), prompt_idx, dtype=torch.long, device=device),
                torch.arange(class_start_idx, class_start_idx + len(self.class_nodes), device=device)
            ], dim=1).T

            all_edges.append(graph_edges)
            all_edges.append(class_edges)

            prompt_indices.append(prompt_idx)
            class_indices.append(
                torch.arange(class_start_idx, class_start_idx + len(self.class_nodes), device=device))

        # Concatenate all node embeddings
        full_x = torch.cat([data.x, prompt_nodes, class_nodes], dim=0)

        # Concatenate all edges
        new_edge_index = torch.cat(all_edges, dim=1)

        # Create new graph data object
        new_data = Data(x=full_x, edge_index=new_edge_index, batch=data.batch)

        return new_data, class_indices

    def forward(self, data):
        """
        Forward 过程：
        1. 构造新的 Prompt Graph
        2. 经过 GCN 处理
        3. 提取 Prompt Nodes（每个图的索引）
        4. 使用 `global_mean_pool` 进行图分类
        """
        prompted_data, class_indices = self.construct_prompted_graph(data)
        edge_index = to_undirected(prompted_data.edge_index)  # 无向图

        # GCN 处理
        h = self.gcn(prompted_data.x, edge_index)

        # 获取 Prompt Nodes 的表示
        class_embeddings = h[torch.cat(class_indices)]
        # print(class_embeddings[0][:5])
        # print(class_embeddings[1][:5])
        # 分类
        class_preds = self.fc(class_embeddings)

        return F.log_softmax(class_preds, dim=-1)


class ProDIGY(nn.Module):
    def __init__(self, in_feats, out_feats, num_classes=2):
        """
        1. **第一层 GNN（Data Graph）**：学习原始数据节点表示
        2. **第二层 GNN（Task Graph）**：让 `Support Nodes` 通过 `Class Nodes` 进行分类
        3. **预训练阶段（Neighbor Matching 任务）**
        4. **Few-shot 任务学习（训练阶段无 Query Nodes）**
        5. **Zero-shot 任务阶段（使用 Query Nodes 进行推理）**
        """
        super(ProDIGY, self).__init__()

        # **第一层 GNN：Data Graph**
        self.data_gnn = GCNEncoder(in_feats, 128, out_feats)

        # **第二层 GNN：Task Graph**
        self.task_gnn = GCNConv(out_feats, out_feats)

        # **类别节点（Class Nodes）是可训练参数**
        self.class_nodes = nn.Parameter(torch.randn(num_classes, out_feats))

        # **最终分类层**
        self.fc = nn.Linear(out_feats, num_classes)

    def construct_task_graph(self, data, graph_embeds, support_labels, use_query=False):
        """
        **构造任务图（Task Graph）**
        - **训练时（Support Set）**：每个 `Support Node` 仅连接到它正确的 `Class Node`
        - **测试时（Query Set）**：每个 `Query Node` 连接到所有 `Class Nodes`
        """
        num_graphs = data.num_graphs
        device = data.x.device

        # **获取类别节点（Class Nodes）**
        class_nodes = self.class_nodes.to(device)

        # **支持集（Support Set）的索引**
        support_indices = torch.arange(num_graphs, device=device)

        # **训练时，Support Set 连接到正确的 Class Node**
        support_class_edges = torch.stack([
            support_indices,  # Support Nodes (Graph Embeddings)
            support_labels  # Class Nodes (类别索引从 0 开始)
        ], dim=0)

        # **测试时，Query Nodes 连接到所有 Class Nodes**
        if use_query:
            query_indices = support_labels
            query_class_edges = torch.stack([
                query_indices.repeat_interleave(class_nodes.shape[0]),
                torch.arange(class_nodes.shape[0], device=device).repeat(len(query_indices)).clone()
            ], dim=0)
            new_edge_index = query_class_edges
        else:
            query_indices = support_indices  # 训练时，Query Set = Support Set
            # query_class_edges = torch.empty((2, 0), dtype=torch.long, device=device)  # 训练阶段不需要 Query 连接
            new_edge_index = support_class_edges

        # **拼接所有节点（Graph Embeddings + Class Nodes）**
        full_x = torch.cat([graph_embeds, class_nodes], dim=0)

        return full_x, new_edge_index, query_indices

    def forward(self, data, support_labels, use_query=False):
        """
        **前向传播**
        1. **Data Graph 处理原始数据**
        2. **Task Graph 进行 Few-shot 任务学习**
        3. **训练时，使用 Support Set**
        4. **测试时，使用 Query Set**
        """
        edge_index = to_undirected(data.edge_index)

        # **第一层 GNN：处理 Data Graph**
        h = self.data_gnn(data.x, edge_index)

        # **全局池化（Graph Embedding）**
        graph_embeds = global_add_pool(h, data.batch)

        # **构造任务图**
        full_x, task_edge_index, query_indices = self.construct_task_graph(data, graph_embeds, support_labels, use_query=use_query)
        task_edge_index = to_undirected(task_edge_index)

        # **第二层 GNN：处理 Task Graph**
        h = self.task_gnn(full_x, task_edge_index)

        # **获取 Query Nodes**
        query_embeds = h[query_indices]

        # **计算类别预测概率**
        graph_logits = self.fc(query_embeds)

        return F.log_softmax(graph_logits, dim=-1)

    def neighbor_matching_loss(self, h, pos_pairs, neg_pairs):
        """
        **计算 Neighbor Matching 预训练损失**
        - `pos_pairs`: (i, j) 对，表示同一邻域的节点
        - `neg_pairs`: (i, j) 对，表示不同邻域的节点
        """
        # **计算正样本对的余弦相似度**
        pos_sim = F.cosine_similarity(h[pos_pairs[:, 0]], h[pos_pairs[:, 1]], dim=-1)

        # **计算负样本对的余弦相似度**
        neg_sim = F.cosine_similarity(h[neg_pairs[:, 0]], h[neg_pairs[:, 1]], dim=-1)

        # **使用对比损失**
        loss = -torch.mean(torch.log(torch.sigmoid(pos_sim - neg_sim)))

        return loss

class PLAN(nn.Module):
    def __init__(self, embed_dim=300, n_heads=8, n_layers=12, num_classes=2, dropout=0.3, use_time_embed=False,
                 time_bins=100):
        super(PLAN, self).__init__()
        self.embed_dim = embed_dim
        self.use_time_embed = use_time_embed

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=2 * embed_dim,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Time delay embedding（可选）
        if use_time_embed:
            self.time_embed = nn.Embedding(time_bins, embed_dim)
        else:
            self.time_embed = None

        # Attention参数
        self.gamma = nn.Parameter(torch.randn(embed_dim))
        # 分类层
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, tweet_reps, time_bins=None):
        """
        tweet_reps: (B, N, D)  # B: batch, N: posts, D: embed_dim
        time_bins: (B, N)      # 可选，每条tweet的时间延迟分桶

        1. 每条推文（如max-pool过词向量） -> tweet_reps输入
        2. 可选加时间延迟embedding
        3. transformer做self-attention，编码所有tweet间交互
        4. attention池化得整体表示
        5. 分类预测
        """

        # 加时间延迟embedding（可选）
        if self.use_time_embed and time_bins is not None:
            tweet_reps = tweet_reps + self.time_embed(time_bins)

        # transformer编码推文级关系
        post_out = self.transformer_encoder(tweet_reps)  # (B, N, D)

        # Attention池化（论文Eq.3-4）
        # 计算每个post的权重（α），对输出做加权求和，得整体表示v
        attn_logits = torch.matmul(post_out, self.gamma)  # (B, N)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, N)
        v = torch.sum(attn_weights.unsqueeze(-1) * post_out, dim=1)  # (B, D)

        # 分类
        logits = self.fc(v)  # (B, num_classes)
        return logits, attn_weights
