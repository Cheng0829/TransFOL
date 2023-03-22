import torch, time, pickle
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

class PredictionMetrics:
    def __init__(self):
        self._ranks = []
        self._weights = []
        self.pred_category = []
        self.truth_category = []

    def list_10000(self, test_answers):
        answers_list_10000 = []
        for query in test_answers:
            answers_list_10000.extend(test_answers[query])
        # print(len(answers_list_10000))
        queries_list_10000 = []
        for query in test_answers:
            for j in range(len(test_answers[query])):
                queries_list_10000.append(query)
        # print(len(queries_list_10000))
        return answers_list_10000, queries_list_10000


    def digest(self, pred: torch.Tensor, truth: torch.Tensor, weight: torch.Tensor = None,\
         multi_label=True, cjk_queries=None, all_queries=None, answers_num=0, queries_num=None):

        self.pred = pred
        self.truth = truth
        # self.pred_multi_label = self.pred.cpu()
        # 概率小数, 用于roc_auc_score
        # self.cjk_pos_pred_multi_label_softmax = torch.nn.functional.softmax(self.pred_multi_label, dim=1).cpu()
        # self.pred_multi_label[self.pred_multi_label < 0] = 0  # size: [10,61]
        # self.pred_multi_label[self.pred_multi_label > 0] = 1
        discrepancy_list, num_list = [], []
        if cjk_queries != None:
            self.multi_label_flag = 1
        else:
            self.multi_label_flag = 0
        
        if self.multi_label_flag:
            with open(r'F:\kg-datasets\KG_data\FB15k-237-q2b\train-answers.pkl', 'rb') as f:
                train_answers = pickle.load(f)
                train_answers_dict = train_answers
            with open(r'F:\kg-datasets\KG_data\FB15k-237-q2b\test-easy-answers.pkl', 'rb') as f:
                test_easy_answers = pickle.load(f)
            with open(r'F:\kg-datasets\KG_data\FB15k-237-q2b\test-hard-answers.pkl', 'rb') as f:
                test_hard_answers = pickle.load(f)

            import ast
            with open(r'F:\kg-datasets\KG_data\FB15k-237-q2b\effect_reverse_dict.txt', 'r') as f:
                a = f.readlines()
                conc_dict = ast.literal_eval(a[0])
            with open(r'F:\kg-datasets\KG_data\FB15k-237-q2b\effect_dict.txt', 'r') as f:
                effect = f.readlines()
                effect_dict = ast.literal_eval(effect[0])
                num_effect = len(effect_dict)
            # num_effect = 14429
            test_answers = test_easy_answers and test_hard_answers


        # num_effect = 636

        # pred: 激活函数为LeakyReLU(), 矩阵值为正数则代表预测为是,否则代表预测为不是
        num_nodes = pred.shape[0]
        assert truth.shape[0] == num_nodes
        pred_idx = np.argmax(pred.cpu().numpy(), axis=1) # 预测每一行的最大值的索引
        # pred_prob = np.amax(pred, axis=1) # 预测的最大值的具体值
        self.pred_category.extend(list(pred_idx))
        self.truth_category.extend(list(truth.cpu().numpy()))
        truth_prob = pred.gather(dim=1, index=truth.unsqueeze(1))
        rank = pred.gt(truth_prob).sum(dim=1) + 1
        self._ranks.append(rank)
        if weight is None:
            self._weights.append(torch.ones_like(rank))
        else:
            self._weights.append(weight)

        """
        **********************************************************************************************************************
        **********************************************************************************************************************
        **********************************************************************************************************************
        """

        """
        idea1: 由于存在语义关系, 模型可能预测出许多相似的项, 因此遍历每个查询,只把最大的n个值设为1(n为训练集答案数+1) -> 专用于测AUC
        idea2: 应当把pred中的训练集答案赋值为0, 去除不公平性. -> 专用于测hits@n
        idea3: 改一下cjk_pred_multi_label设为0和1的阈值
        """
        if self.multi_label_flag:
            # print('self.pred.shape', self.pred.shape)
            # print(self.truth[:64])
            answers_list_10000, queries_list_10000 = self.list_10000(test_answers)
            self.cjk_pred_multi_label = self.pred[:,:num_effect].cpu()
            """
            接下来仅需要找到truth矩阵,也就是先遍历查询,然后找到其在数据集中的答案,将对应位置赋值为1
            由于大小确定, 直接遍历查询列表即可,有几个答案就遍历几行
            """
            self.cjk_truth_multi_label = torch.zeros(self.cjk_pred_multi_label.shape).cpu()
            self.int_cjk_pred_multi_label = torch.zeros(self.cjk_pred_multi_label.shape).cpu()
            # queries_num = 0 # queries_num对整体的数据集答案进行遍历
            discrepancy_list = []
            num_list = []
            for i in range(len(self.cjk_pred_multi_label)):
                query = queries_list_10000[queries_num+i]
                train_answers = list(train_answers_dict[query])
                # print(train_answers)
                test_answer = answers_list_10000[i]
                self.cjk_truth_multi_label[i][train_answers] = 1
                if len(train_answers) == 0 or 1==1:
                    self.cjk_truth_multi_label[i][test_answer] = 1

                '''idea1: 由于存在语义关系, 模型可能预测出许多相似的项, 因此遍历每个查询,只把最大的n个值设为1(n为训练集答案数+1) -> 专用于测AUC'''
                train_answer_num = len(train_answers)
                # print(train_answer_num)
                tmp = np.zeros(num_effect)
                indices = np.argsort(np.array(self.cjk_pred_multi_label[i]))[::-1] # 降序排序后的原下标
                # print(indices)
                for j in range(train_answer_num+1):
                    # print(indices[i], self.cjk_pred_multi_label[i][indices[i]])
                    # tmp[indices[j]] = 1  # 只保留前n+1个最大值
                    tmp[indices[j]] = self.cjk_pred_multi_label[i][indices[j]]
                    self.int_cjk_pred_multi_label[i][indices[j]] = 1
                # print(tmp)
                # print(self.cjk_pred_multi_label[i])
                # print(tmp.shape, self.cjk_pred_multi_label[i].shape)
                self.cjk_pred_multi_label[i] = torch.tensor(tmp)
                # print(self.cjk_pred_multi_label[i])
                # print(self.cjk_pred_multi_label[i].cuda())
                # time.sleep(90)
                
                '''idea2: 应当把pred中的训练集答案赋值为0, 去除不公平性. -> 专用于测hits@n'''
                # self.cjk_pred_multi_label[i][train_answers] = 0
                
                """剂量实验:pred与truth浓度之差"""
                try:
                    conc_truth = train_answers
                    if test_answer not in conc_truth:
                        conc_truth.append(test_answer)
                    conc_pred = indices[:1] # 只取一个
                    # conc_pred = indices[:min(3,len(conc_truth))] # 只取3个
                    # conc_pred = indices[:len(conc_truth)]
                    conc_pred_list = []
                    conc_truth_list = []
                    for i in range(len(conc_truth)):
                        conc_truth_list.append(float(conc_dict[conc_truth[i]]))
                    for i in range(len(conc_pred)):
                        conc_pred_list.append(float(conc_dict[conc_pred[i]]))
                    discrepancy = abs(np.mean(conc_pred_list) - np.mean(conc_truth_list))
                    import  math
                    if math.isnan(discrepancy) == False:
                        discrepancy_list.append(discrepancy)
                except:
                    pass
            # print('\n差值为:%f\n'%np.sum(discrepancy_list))
            # print('\n个数为:%f\n'%len(discrepancy_list))

            queries_num = queries_num + len(self.cjk_pred_multi_label)
            answers_num = queries_num

            '''二值化'''
            # self.cjk_pred_multi_label[self.cjk_pred_multi_label>0] = 1
            # self.cjk_pred_multi_label[self.cjk_pred_multi_label<0] = 0
            # print('queries_num', queries_num)

            '''pred只保留最大值'''
            # indices = np.argmax(self.cjk_pred_multi_label, axis=1)
            # indices = np.expand_dims(indices, axis=1)
            # output = np.zeros_like(self.cjk_pred_multi_label)
            # np.put_along_axis(output, indices, 1, axis=1)
            # self.cjk_pred_multi_label = torch.tensor(output)
            # 概率小数, 用于roc_auc_score
            self.cjk_pos_pred_multi_label_softmax = torch.nn.functional.softmax(self.cjk_pred_multi_label, dim=1).cpu()
        
        # print('\ncjk'+200*'*'+'cjk\n')
        if self.multi_label_flag:
            self.cjk_pred_multi_label = self.cjk_pred_multi_label.cuda()
            truth = truth.cuda()
            # pred: 激活函数为LeakyReLU(), 矩阵值为正数则代表预测为是,否则代表预测为不是
            num_nodes =self.cjk_pred_multi_label.shape[0]
            assert truth.shape[0] == num_nodes
            pred_idx = np.argmax(self.cjk_pred_multi_label.cpu().numpy(), axis=1) # 预测每一行的最大值的索引
            # pred_prob = np.amax(pred, axis=1) # 预测的最大值的具体值
            self.pred_category.extend(list(pred_idx))
            self.truth_category.extend(list(truth.cpu().numpy()))
            truth_prob = self.cjk_pred_multi_label.gather(dim=1, index=truth.cuda().unsqueeze(1)).cuda()
            rank = self.cjk_pred_multi_label.gt(truth_prob.cuda()).sum(dim=1) + 1
            self._ranks.append(rank)
            if weight is None:
                self._weights.append(torch.ones_like(rank).cuda())
            else:
                self._weights.append(weight)
        
        return cjk_queries, answers_num, queries_num, np.sum(discrepancy_list), len(discrepancy_list)

    def get_ranks(self): # 返回真实结果在预测中的排名

        if len(self._ranks) > 1:
            self._ranks = [torch.cat(self._ranks)]
        return self._ranks[0]

    def get_weight(self):
        w = torch.cat(self._weights)
        self._weights = [w]
        return w

    def _weighted_mean(self, arr):
        w = self.get_weight()
        return ((arr.float() * w).sum() / w.sum()).item()

    r"""
    The definitions of the metrics are taken from
    https://github.com/sebastianruder/NLP-progress/blob/master/english/relation_prediction.md#metrics
    """

    def MRR(self): # KG_Predict: 0.261
        return self._weighted_mean(self.get_ranks().float().reciprocal()) # reciprocal倒数
    def hits_at(self, k): # KG_Predict: Hits@1:0.174, Hits@3:0.266, Hits@10:0.447 
        return self._weighted_mean(self.get_ranks().le(k))
    def macro_f1_score(self):
        """
        4.KG4SL: 0.8877(F1,暂不知是哪一种F1)
        """
        # return -1
        if self.multi_label_flag == 1:
            f1_mean = []
            for i in range(self.int_cjk_pred_multi_label.size(0)):
                # print(i)
                # print(self.truth_multi_label[i], self.pred_multi_label[i])
                f1 = metrics.f1_score(self.cjk_truth_multi_label[i].cpu(), self.int_cjk_pred_multi_label[i].cpu(), average='macro') # Macro-F1: 先算每个类别的F1, 再求平均
                f1_mean.append(f1) 
            return np.mean(f1_mean)
        else:
            return metrics.f1_score(self.truth_category, self.pred_category, average='macro') # Macro-F1: 先算每个类别的F1, 再求平均
    def micro_f1_score(self):
        """
        marco-F1：先计算每一类下F1值，最后求和做平均值就是macro-F1， 这种情况就是不考虑数据的数量，平等的看每一类；
        micro-F1： 是当二分类计算，通过计算所有类别的总的Precision和Recall，然后计算出来的F1值即为micro-F1；
        泛化1p+2p=pi实验中:不同数据类别数据量极不均衡,因此应看micro-F1
        """
        # return -1
        if self.multi_label_flag == 1:
            f1_mean = []
            for i in range(self.int_cjk_pred_multi_label.size(0)):
                f1 = metrics.f1_score(self.cjk_truth_multi_label[i].cpu(), self.int_cjk_pred_multi_label[i].cpu(), average='micro') # Micro-F1: 直接计算总体的F1
                f1_mean.append(f1)
            return np.mean(f1_mean)
        else:
            return metrics.f1_score(self.truth_category, self.pred_category, average='micro') # Micro-F1: 直接计算总体的F1
    def ROC_AUC(self): 
        """
        1.MSTE: 0.9744
        2.MUFFIN: 0.9160
        3.SumGNN: 0.9465
        4.KG4SL: 0.9470
        10.SGCL-DTI: 0.9438
        """
        if self.multi_label_flag == 1:
            ROC_AUC_mean = []
            for i in range(self.cjk_pred_multi_label.size(0)):
                try: 
                    """多标签的train_answers个数应>1, truth_answer就应>2"""
                    if np.sum(np.array(self.cjk_truth_multi_label[i].cpu())>0) > 4 or 1==0:
                        ROC_AUC_score = metrics.roc_auc_score(self.cjk_truth_multi_label[i].cpu(), self.cjk_pred_multi_label[i].cpu())
                        # ROC_AUC_score = metrics.roc_auc_score(self.cjk_truth_multi_label[i].cpu(), self.cjk_pos_pred_multi_label_softmax[i].cpu())
                        ROC_AUC_mean.append(ROC_AUC_score)
                except:
                    pass
            return np.mean(ROC_AUC_mean)
        else:
            return -1

    def PR_AUC(self): 
        """
        1.MSTE: 0.9673
        2.MUFFIN: 0.7033
        3.SumGNN: 0.9321
        4.KG4SL: 0.9564
        10.SGCL-DTI: 0.9578
        """
        if self.multi_label_flag == 1:
            PR_AUC_mean = []
            for i in range(self.cjk_pred_multi_label.size(0)):
                try:
                    # PR_AUC_score = metrics.average_precision_score(self.cjk_truth_multi_label[i].cpu(), self.cjk_pred_multi_label[i].cpu())
                    if np.sum(np.array(self.cjk_truth_multi_label[i].cpu())>0) > 4 or 1==0:
                        precision, recall, thereshold = metrics.precision_recall_curve(self.cjk_truth_multi_label[i].cpu(), self.cjk_pred_multi_label[i].cpu())
                        PR_AUC_score = metrics.auc(recall, precision)
                        PR_AUC_mean.append(PR_AUC_score)
                except:
                    pass
            return np.mean(PR_AUC_mean)
        else:
            return -1

def loss_cross_entropy_multi_ans(score, query, ans, posi_x, posi_ans, query_w=None):
    assert len(posi_x) >= len(query)
    device = score.device
    num_nodes = len(score)
    score = score.exp()
    ent_posi_sum = torch.zeros(num_nodes, dtype=torch.double, device=device)
    from deter_util import deter_scatter_add_
    deter_scatter_add_(posi_x, score[posi_x, posi_ans], ent_posi_sum)
    ans_score = score[query, ans]
    ans_score[ans_score < 0] = 1e-10
    loss_arr = ans_score / (score.sum(dim=1)[query] - ent_posi_sum[query] + ans_score)
    loss_arr = -loss_arr.log()
    # assert all((score.sum(dim=1)[query] - ent_posi_sum[query] + ans_score) > 0)
    if query_w is None:
        query_w = torch.ones_like(loss_arr)
    loss = torch.sum(loss_arr * query_w)
    weight_sum = query_w.sum()
    return loss.float(), weight_sum

def loss_label_smoothing_multi_ans(score, query, ans, posi_x, posi_ans, smoothing, query_w=None):
    assert len(posi_x) >= len(query)
    device = score.device
    num_nodes = len(score)
    score = score.exp()
    ent_posi_sum = torch.zeros(num_nodes, dtype=torch.double, device=device)
    from deter_util import deter_scatter_add_
    deter_scatter_add_(posi_x, score[posi_x, posi_ans], ent_posi_sum)
    ans_score = score[query, ans]
    ans_score[ans_score < 0] = 1e-10
    loss_arr = ans_score / (score.sum(dim=1)[query] - ent_posi_sum[query] + ans_score)
    whole_loss = score[query] / (score.sum(dim=1)[query] - ent_posi_sum[query] + ans_score).unsqueeze(1)
    # assert all(score.sum(dim=1)[query] - ent_posi_sum[query] + ans_score > 0)
    whole_loss = -whole_loss.log()
    loss_arr = -loss_arr.log()
    if query_w is None:
        query_w = torch.ones_like(loss_arr)
    loss = torch.sum(loss_arr * query_w)
    rand_loss = torch.sum(whole_loss.mean(-1) * query_w)
    weight_sum = query_w.sum()
    LSloss = smoothing * rand_loss + (1 - smoothing) * loss
    return LSloss, weight_sum
