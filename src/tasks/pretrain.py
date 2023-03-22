import torch

from graph_util import Graph


class FB15K237:
    def __init__(self, config):
        from .betae import BetaEDataset
        self.betae = BetaEDataset(config['data_dir'])
        """self.train_edge: [[10, 0, 1], [5, 0, 1], [6, 0, 1]......]"""
        self.train_edge, self.valid_edge, self.test_edge = self.betae.calldata()
        self.num_nodes = 0
        self.relation_cnt = 0

        def edge_sanitize(h, r, t):
            """h是头实体, r是关系, t是尾实体"""
            if r % 2 == 1:
                h, t = t, h
                r = int((r - 1) / 2)
                return None  # The edge is added in the inverse counterpart
            else:
                r = int(r / 2)
            """按照训练集实体序号确定超参数, 所以拿处理后的数据集作为训练集, 可能越界"""
            if h >= self.num_nodes:
                self.num_nodes = h + 1
            if t >= self.num_nodes:
                self.num_nodes = t + 1
            if r >= self.relation_cnt:
                self.relation_cnt = r + 1
            return h, r, t

        def batch_edge_san(arr):
            for t in arr:
                t = edge_sanitize(*t)
                if t is not None:
                    yield t

        self.train_edge = list(batch_edge_san(self.train_edge)) # [272115, 3]
        self.valid_edge = list(batch_edge_san(self.valid_edge))
        self.test_edge = list(batch_edge_san(self.test_edge))

    def get_full_train_graph(self):
        device = torch.device('cpu')
        arr = torch.tensor(self.train_edge, device=device).T
        # print(arr.shape) # [3, 272115]
        # print(arr[[0, 2]].shape) # [2, 272115]
        # print(arr[1].shape) # [272115]
        return Graph(
            x=torch.arange(self.num_nodes, device=device), # torch.arange(5)  -> tensor([ 0,  1,  2,  3,  4])
            edge_index=arr[[0, 2]],
            edge_attr=arr[1],
        )

    def dataloader_train(self, config):
        from data_util import dataloader_pretrain
        # print(self.num_nodes,self.relation_cnt) 
        # print(len(dataloader_pretrain(self.get_full_train_graph(), self.num_nodes, self.relation_cnt, config)))
        return dataloader_pretrain(
            self.get_full_train_graph(),
            self.num_nodes, # 14505
            self.relation_cnt, # 237
            config,
        )

    def dataloader_test(self):
        from data_util import dataloader_test
        return dataloader_test(
            self.test_edge,
            self.train_edge + self.valid_edge + self.test_edge,
            self.num_nodes,
            self.relation_cnt,
        )
