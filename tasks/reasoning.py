import torch

from graph_util import GraphWithAnswer


def query_to_graph(mode, q, hard_ans_list, mask_list=None, gen_pred_mask=True, device=torch.device('cpu')):
    r"""
    Assert that easy_ans_set have included all nodes to be masked
    """
    if not gen_pred_mask:
        assert mask_list is None or len(hard_ans_list) == len(mask_list)
    x = []
    edge_ans = []

    edge_index = [[], []]
    edge_attr = []

    def add_raw_edge(a, r, b):
        """
        add_raw_edge(0, q[0][1][0], 2) # add_raw_edge(0, 1, 2)
        add_raw_edge(1, q[1][1][0], 2) # add_raw_edge(1, 1, 2)
        """
        # add_raw_edge第一个参数是关系边头实体的编号, 第二个参数是关系边的在字典中的编号, 三个是关系边头实体的编号
        if r % 2 == 1:
            a, b = b, a
            r = (r - 1) / 2
        else:
            r = r / 2
        edge_index[0].append(a)
        edge_index[1].append(b)
        edge_attr.append(r)

    q_cnt = 0
    x_query = []
    x_ans = []
    x_pred_weight = []
    x_pred_mask_x = []
    x_pred_mask_y = []
    joint_nodes = []
    union_query = []
    def push_anslist(node_id, hard_anslist, mask_list):
        nonlocal x_query, x_ans, x_pred_weight
        anslen = len(hard_anslist)
        if anslen == 0:
            return
        x_query += [node_id] * anslen
        x_ans += hard_anslist
        x_pred_weight += [1 / anslen] * anslen
        if not gen_pred_mask:
            return
        nonlocal x_pred_mask_x, x_pred_mask_y, q_cnt
        # assert that mask_list already contains hard_anslist
        x_pred_mask_x += [node_id] * len(mask_list)
        x_pred_mask_y += mask_list

    mask_list = mask_list or []
    if mode == '1p':
        x = [q[0], -1]
        add_raw_edge(0, q[1][0], 1)
        push_anslist(1, hard_ans_list, mask_list)
    elif mode == '2p':
        r"""
        0 - 1(-1) - 2(-1)
        """
        x = [q[0], -1, -1]
        add_raw_edge(0, q[1][0], 1)
        add_raw_edge(1, q[1][1], 2)
        push_anslist(2, hard_ans_list, mask_list)
    elif mode == '2i':
        r"""
        0 - 
            2(-1)
        1 - 
        """
        '''每个样本运行一次query_to_graph,一个批共运行batch_size次'''
        # print('q:',q)
        # q即为query batch_size=4, 则q: ((10, (1,)), (1, (1,))) ......, 每次为1个2i查询
        # print(q)
        x = [q[0][0], q[1][0], -1] # x = [10,1,-1]
        # 2i, 两条关系边, 所以用两次add_raw_edge
        # add_raw_edge第一个参数是关系边头实体的编号, 第二个参数是关系边的在字典中的编号, 三个是关系边头实体的编号
        add_raw_edge(0, q[0][1][0], 2) # add_raw_edge(0, 1, 2)
        add_raw_edge(1, q[1][1][0], 2) # add_raw_edge(1, 1, 2)
        # push_anslist第一个参数是当前的关系边的数量
        push_anslist(2, hard_ans_list, mask_list)
    elif mode == 'pi':
        r"""
        0 - 1(-1) -
                    3(-1)
                2 -  
        """
        x = [q[0][0], -1, q[1][0], -1]
        add_raw_edge(0, q[0][1][0], 1)
        add_raw_edge(1, q[0][1][1], 3)
        add_raw_edge(2, q[1][1][0], 3)
        push_anslist(3, hard_ans_list, mask_list)

    g = GraphWithAnswer(
        x=torch.tensor(x, device=device, dtype=torch.long),
        edge_index=torch.tensor(edge_index, device=device, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, device=device, dtype=torch.long),
        x_query=torch.tensor(x_query, device=device, dtype=torch.long),
        x_ans=torch.tensor(x_ans, device=device, dtype=torch.long),
        edge_ans=torch.tensor(edge_ans, device=device, dtype=torch.long),
        x_pred_weight=torch.tensor(x_pred_weight, device=device, dtype=torch.float),
        joint_nodes=torch.tensor(joint_nodes, device=device, dtype=torch.long),
        union_query=torch.tensor(union_query, device=device, dtype=torch.long),
    )

    if gen_pred_mask:
        g.x_pred_mask = torch.tensor([x_pred_mask_x, x_pred_mask_y], device=device, dtype=torch.long)
    return g


from .betae import BetaEDataset


class FB15K237_reasoning:
    def __init__(self, betae: BetaEDataset, relation_cnt, train_mode, test_mode): # test_modes为2i
        super(FB15K237_reasoning, self).__init__()
        self.betae = betae
        self.train_mode = train_mode
        self.test_mode = test_mode
        self.relation_cnt = relation_cnt

        self.train_query = betae.get_file("train-queries.pkl")
        self.train_answer = betae.get_file("train-answers.pkl")

    @staticmethod
    def _batch_q2g(batch, relation_cnt):
        cpu = torch.device('cpu')
        from graph_util import MatGraph, BatchMatGraph
        arr = [query_to_graph(*x, device=cpu) for x in batch]
        arr = [MatGraph.make_line_graph(g, relation_cnt) for g in arr]
        return BatchMatGraph.from_mat_list(arr)

    def _get_test_dataloader(self, query, pred_answer, mask_answer, modelist, gen_pred_mask, batch_size, num_workers):
        data = []
        hard_answer = pred_answer
        easy_answer = mask_answer
        for m in modelist: # 2i
            for q in query[m]:
                mask_set = hard_answer[q]
                if easy_answer is not None: 
                    # mask_set = hard_answer[q] | easy_answer[q] # 集合求并集
                    for tmp in easy_answer[q]:
                        if tmp not in hard_answer[q]:
                            mask_set.append(tmp)
                data.append((m, q, list(hard_answer[q]), list(mask_set), gen_pred_mask))
        
        '''在TWOSIDES-10重复中,data和query均为10'''
        from torch.utils.data import DataLoader
        from data_util import SubsetSumSampler
        from functools import partial
        # print(data)
        batch_sampler = SubsetSumSampler(
            [len(x[2]) for x in data], # x[2]即hard_answer[q]: [1,1,1,...]
            lim=batch_size,
        )
        data = DataLoader(
            data,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                FB15K237_reasoning._batch_q2g,
                relation_cnt=self.relation_cnt,
            ),
            num_workers=num_workers,
        )
        '''
        collate_fn的用处:
            自定义数据堆叠过程
            自定义batch数据的输出形式
        collate_fn的使用:
            定义一个以data为输入的函数
            注意, 输入输出分别域getitem函数和loader调用时对应
        '''
            
        return data

    def dataloader_fine_tune(self, config):
        mode = self.train_mode
        ans = self.train_answer
        return self._get_test_dataloader(self.train_query, pred_answer=ans, mask_answer=None,
                                         modelist=mode, gen_pred_mask=False,
                                         batch_size=config['batch_size'],
                                         num_workers=config['num_workers'])

    def dataloader_test(self, config):
        # test_query长度为10
        test_query = self.betae.get_file("test-queries.pkl")
        test_hard_answer = self.betae.get_file("test-hard-answers.pkl")
        test_easy_answer = self.betae.get_file("test-easy-answers.pkl")

        mode = self.test_mode
        hardans = test_hard_answer
        easyans = test_easy_answer
        query = test_query
        
        dataloader = self._get_test_dataloader(query, pred_answer=hardans, mask_answer=easyans, modelist=mode,
                                         gen_pred_mask=True, batch_size=config['batch_size'],
                                         num_workers=config['num_workers'])

        return dataloader # 乱序的7971
 
    def dataloader_valid(self, config):
        valid_query = self.betae.get_file("valid-queries.pkl")
        valid_hard_answer = self.betae.get_file("valid-hard-answers.pkl")
        valid_easy_answer = self.betae.get_file("valid-easy-answers.pkl")
        mode = self.test_mode
        hardans = valid_hard_answer
        easyans = valid_easy_answer
        query = valid_query
        return self._get_test_dataloader(query, pred_answer=hardans, mask_answer=easyans, modelist=mode,
                                         gen_pred_mask=True, batch_size=config['batch_size'],
                                         num_workers=config['num_workers'])
