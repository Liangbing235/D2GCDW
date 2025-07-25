# 基于思想的视图缺失补全模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv  # 使用 GCN 或 GAT 卷积层

class AdaptiveTagEncoding(nn.Module):
    def __init__(self, num_views=6):
        super(AdaptiveTagEncoding, self).__init__()
        self.num_views = num_views
        self.tag_embedding = nn.Embedding(2 ** num_views, num_views)  # 2^6 = 64 possible combinations

    def forward(self, missing_pattern):
        """
        missing_pattern: tensor of shape (batch_size,) representing the missing pattern
        """
        tags = self.tag_embedding(missing_pattern)  # (batch_size, num_views)
        return tags

class CommonSpaceProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CommonSpaceProjection, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, use_gat=False):
        super(GraphNeuralNetwork, self).__init__()
        self.use_gat = use_gat
        self.convs = nn.ModuleList()
        if use_gat:
            self.convs.append(GATConv(input_dim, hidden_dim, heads=8, dropout=0.6))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim * 8, hidden_dim, heads=1, concat=False, dropout=0.6))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.fc(x)
        return x

class GNNViewCompletionModule(nn.Module):
    def __init__(self, num_views, d_model=128, input_dims=None, gnn_hidden_dim=64, gnn_output_dim=128, use_gat=True):
        super(GNNViewCompletionModule, self).__init__()
        self.num_views = num_views
        self.d_model = d_model
        self.use_gat = use_gat

        if input_dims is not None:
            self.view_transformers = nn.ModuleList([nn.Linear(input_dim, d_model) for input_dim in input_dims])
        else:
            raise ValueError("input_dims must be provided to initialize view transformers.")

        # 初始化 GNN 模块
        self.gnn = GraphNeuralNetwork(input_dim=d_model, hidden_dim=gnn_hidden_dim, output_dim=gnn_output_dim, use_gat=True)

    def forward(self, X, missing_pattern):
        transformed_views = [self.view_transformers[v](X[v]) for v in range(self.num_views)]
        mask = self._create_mask(missing_pattern)

        # 构建图结构
        edge_index = self._build_graph(mask)

        # 将所有视图堆叠成一个图节点特征矩阵
        node_features = torch.cat(transformed_views, dim=0)  # (num_views * batch_size, d_model)

        # 通过 GNN 模型
        completed_node_features = self.gnn(node_features, edge_index)

        # 将完成的特征重新分割为每个视图
        completed_views = torch.split(completed_node_features, X[0].size(0), dim=0)  # [(batch_size, d_model)] * num_views

        # 将列表转换为张量
        completed_views = torch.stack(completed_views, dim=1)  # (batch_size, num_views, d_model)
        return completed_views

    def _create_mask(self, missing_pattern):
        mask = torch.ones((missing_pattern.size(0), self.num_views)).to(missing_pattern.device)
        for i in range(missing_pattern.size(0)):
            pattern = missing_pattern[i].item()
            for v in range(self.num_views):
                if (pattern >> v) & 1 == 0:
                    mask[i, v] = 0
        return mask

    def _build_graph(self, mask):
        """
        构建图结构，根据 mask 生成边索引
        """
        batch_size, num_views = mask.size()
        edge_index = []

        # 为每个样本构建视图之间的边
        for i in range(batch_size):
            for v1 in range(num_views):
                for v2 in range(v1 + 1, num_views):
                    if mask[i, v1] == 1 and mask[i, v2] == 1:
                        edge_index.append([i * num_views + v1, i * num_views + v2])
                        edge_index.append([i * num_views + v2, i * num_views + v1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(mask.device)
        return edge_index

class GatedFusionModule(nn.Module):
    def __init__(self, d_model):
        super(GatedFusionModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, transformer_output, lhgn_output):
        # 将两个模块的输出拼接起来
        combined = torch.cat([transformer_output, lhgn_output], dim=-1)

        # 计算门控权重
        gate = self.gate(combined)

        # 融合两个模块的输出
        fused = self.fusion(combined)

        # 应用门控权重
        output = gate * transformer_output + (1 - gate) * lhgn_output
        return output

class TMC(nn.Module):
    def __init__(self, classes, views, classifier_dims, input_dims, lambda_epochs=1):
        super(TMC, self).__init__()
        self.views = views
        self.classes = classes
        self.input_dims = input_dims
        self.lambda_epochs = lambda_epochs

        # 每个视图的分类器
        self.Classifiers = nn.ModuleList([
            Classifiers([128, 256, 128, 64], classes=self.classes) for v in range(self.views)
        ])
        self.Classifiers1 = nn.ModuleList([Classifiers1([128, 256, 128, 64], classes=self.classes)])

        # 为每个视图添加一个线性层，将输入特征维度映射到统一的 128 维
        self.view_transformers = nn.ModuleList([
            nn.Linear(input_dims[v], 128) for v in range(self.views)
        ])

        # 视图缺失补全模块（Transformer 和 LHGN）
        self.transformer_view_completion = ViewCompletionModule(num_views=views, d_model=128, input_dims=input_dims)
        self.gnn_view_completion = GNNViewCompletionModule(num_views=views, input_dims=input_dims, d_model=128)

        # 门控融合模块
        self.gated_fusion = GatedFusionModule(d_model=128)


    def forward(self, X, target, epoch, missing_pattern):
        # Step 1: 视图缺失补全
        transformer_completed_views = self.transformer_view_completion(X, missing_pattern)  # (batch_size, num_views, 128)
        gnn_completed_views = self.gnn_view_completion(X, missing_pattern)  # (batch_size, num_views, 128)

        # Step 2: 门控融合
        fused_completed_views = []
        for v in range(self.views):
            fused_view = self.gated_fusion(transformer_completed_views[:, v, :], gnn_completed_views[:, v, :])
            fused_completed_views.append(fused_view)
        fused_completed_views = torch.stack(fused_completed_views, dim=1)  # (batch_size, num_views, 128)

        # Step 3: 计算视图补全损失 (MSE Loss)
        mask = self._create_mask(missing_pattern)  # (batch_size, num_views)
        original_views = torch.stack([self.view_transformers[v](X[v]) for v in range(self.views)],
                                     dim=1)  # (batch_size, num_views, 128)
        mse_loss = F.mse_loss(fused_completed_views * mask.unsqueeze(-1), original_views * mask.unsqueeze(-1),
                              reduction='sum') / mask.sum()

        # Step 3: 动态融合
        evidence = self.infer(fused_completed_views)
        alpha = [evidence[v] + 1 for v in range(self.views)]
        h = self.dynamic_Combin(alpha, fused_completed_views)

        # Step 4: 分类 (五分类任务)
        logits = self.Classifiers1[0](h)
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == target).sum().item() / target.size(0)

        # Step 5: 损失计算
        loss = sum([self.ce_loss(target, alpha[v], self.classes, epoch, self.lambda_epochs) for v in range(self.views)]) / self.views
        ce_loss = F.cross_entropy(logits, target)
        l2_reg = sum(torch.norm(param) for param in self.parameters())

        # 总损失
        # total_loss = (0.05 * loss + ce_loss + 0.001 * l2_reg + mse_loss).mean()
        total_loss = (0.05 * loss + ce_loss + 0.001 * l2_reg + 0.5 * mse_loss).mean()

        return evidence, total_loss, accuracy, predicted
        # return h, total_loss, accuracy, predicted

    def _create_mask(self, missing_pattern):
        """
        根据缺失模式创建掩码 (mask)
        """
        mask = torch.ones((missing_pattern.size(0), self.views)).to(missing_pattern.device)
        for i in range(missing_pattern.size(0)):
            pattern = missing_pattern[i].item()
            for v in range(self.views):
                if (pattern >> v) & 1 == 0:  # 该视图缺失
                    mask[i, v] = 0
        return mask

    def infer(self, input):
        evidence = [self.Classifiers[v](input[:, v, :]) for v in range(self.views)]
        return evidence

    def dynamic_Combin(self, alpha, X):
        h = 0
        lambda1 = 0.8
        for v in range(self.views):
            S_v = torch.sum(alpha[v], dim=1, keepdim=True)
            E_v = alpha[v] - 1
            b_v = E_v / S_v
            b_v_square = torch.matmul(b_v, b_v.T)
            diag_elements = torch.diagonal(b_v_square, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            u_v = self.classes / S_v
            w_v = lambda1 * diag_elements / (u_v + 1)
            h_v = X[:, v, :]
            h += w_v * h_v
        return h

    @staticmethod
    def ce_loss(p, alpha, c, global_step, annealing_step):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = F.one_hot(p, num_classes=c)
        A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        annealing_coef = min(1, global_step / annealing_step)
        alp = E * (1 - label) + 1
        B = annealing_coef * TMC.KL(alp, c)
        return (A + B)

    @staticmethod
    def KL(alpha, c):
        beta = torch.ones((1, c)).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

class Classifiers(nn.Module):
    """
    基础分类器
    """
    def __init__(self, classifier_dims, classes):
        super(Classifiers, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(classifier_dims[i], classifier_dims[i + 1]) for i in range(len(classifier_dims) - 1)
        ])
        self.output_layer = nn.Linear(classifier_dims[-1], classes)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.output_layer(h)


class Classifiers1(nn.Module):
    """
    基础分类器
    """
    def __init__(self, classifier_dims, classes):
        super(Classifiers1, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(classifier_dims[i], classifier_dims[i + 1]) for i in range(len(classifier_dims) - 1)
        ])
        self.output_layer = nn.Linear(classifier_dims[-1], classes)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.output_layer(h)

# 视图缺失补全相关模块
class AdaptiveTagEncoding(nn.Module):
    def __init__(self, num_views=6):
        super(AdaptiveTagEncoding, self).__init__()
        self.num_views = num_views
        self.tag_embedding = nn.Embedding(2 ** num_views, num_views)  # 2^6 = 64 possible combinations

    def forward(self, missing_pattern):
        """
        missing_pattern: tensor of shape (batch_size,) representing the missing pattern
        """
        tags = self.tag_embedding(missing_pattern)  # (batch_size, num_views)
        return tags

class CommonSpaceProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CommonSpaceProjection, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

# 视图缺失补全相关模块
class ViewCompletionModule(nn.Module):
    def __init__(self, num_views, d_model=128, input_dims=None):
        super(ViewCompletionModule, self).__init__()
        self.num_views = num_views
        self.d_model = d_model
        self.transformer = TransformerEncoderDecoder(d_model=d_model, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1)

        if input_dims is not None:
            self.view_transformers = nn.ModuleList([nn.Linear(input_dim, d_model) for input_dim in input_dims])
        else:
            raise ValueError("input_dims must be provided to initialize view transformers.")

    def forward(self, X, missing_pattern):
        transformed_views = [self.view_transformers[v](X[v]) for v in range(self.num_views)]
        mask = self._create_mask(missing_pattern)
        masked_views = [transformed_views[v] * mask[:, v].unsqueeze(-1) for v in range(self.num_views)]
        stacked_views = torch.stack(masked_views, dim=1)
        completed_views = self.transformer(stacked_views, stacked_views)  # Transformer encoder-decoder
        return completed_views

    def _create_mask(self, missing_pattern):
        mask = torch.ones((missing_pattern.size(0), self.num_views)).to(missing_pattern.device)
        for i in range(missing_pattern.size(0)):
            pattern = missing_pattern[i].item()
            for v in range(self.num_views):
                if (pattern >> v) & 1 == 0:
                    mask[i, v] = 0
        return mask

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderDecoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, src, tgt):
        """
        src: encoder input (batch_size, seq_length, d_model)
        tgt: decoder input (batch_size, seq_length, d_model)
        """
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output