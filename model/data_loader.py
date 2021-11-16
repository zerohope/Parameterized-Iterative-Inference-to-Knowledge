import json
import torch
from Constant2 import Constants as config
import pandas as pd
import numpy as np
import itertools
import tqdm

# 添加根路径名
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)




# 获取Q，（q_num,kn_num）以及d试题的度矩阵(q_num,1)
def getCommonData_common(root_path):

    # Q矩阵
    Q = pd.read_csv(root_path + "/Q.csv",encoding="ISO-8859-15", low_memory=False )
    # Q = pd.read_csv("../data_set/Q.csv",encoding="ISO-8859-15", low_memory=False)
    # 每个题的度（包含的知识点）
    # d = pd.read_csv("../data_set/q_kn_num.csv",encoding="ISO-8859-15", low_memory=False)
    d = pd.read_csv(root_path + "/q_kn_num.csv",encoding="ISO-8859-15", low_memory=False )
    # 实质为dataFrame转为numpy，在转为tensor
    # Q_tensor = torch.tensor(Q.values)
    Q_tensor = torch.FloatTensor(Q.values)

    # d_tensor = torch.tensor(d.values)
    d_tensor = torch.FloatTensor(d.values)
    # rows = d_tensor.shape[0]
    #
    # d_tensor = d_tensor.resize(rows)

    return Q_tensor,d_tensor

class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self,data_dim,root_path):
        self.root_path = root_path
        self.batch_size = config.train.batch_size
        self.ptr = 0
        self.data = []
        self.q_num = data_dim['exer_n']
        self.knowledge_num = data_dim['knowledge_n']

        data_file = root_path+'/train_data.json'
        # data_file = root_path+'../data_set/train_data.json'
        # config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
            # print('读取数据json。load（）')
            # print(self.data)

        # with open(config_file) as i_f:
        #     i_f.readline()
        #     _, _, knowledge_n = i_f.readline().split(',')
        # self.knowledge_dim = int(knowledge_n)
        self.end_num = 50
        # self.end_num = len(self.data)

    def getCommonData(self):
        Q,d = getCommonData_common(self.root_path)
        return Q,d


            # 接下来 kn_num行数据 使用这种迭代器的方式似乎无法实现  将其用某个特殊符号隔开？ 然后在分？
    #         以数组的形式放在一行 ，见temp read_test()




    def next_batch(self):
        # print('训练数据集大小：',len(self.data))
        print('dataloader:当前第{}条数据'.format(self.ptr + self.batch_size))
        if self.is_end():
            return None, None, None, None
        # input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        stu_id, kn_id, score_emb, user_k_kc, ex_id, labels = [], [], [], [], [], []

        # 一个用户的一条记录
        for count in range(self.batch_size):
            # if self.ptr % 100 == 0:
            #     print('当前第{}条数据：'.format(self.ptr))
            # print('当前第{}条数据：'.format(self.ptr))
            log = self.data[self.ptr + count]
            # print('debug: next_batch() :log: \r\n{}'.format(log))
            # knowledge_emb = [0.] * self.knowledge_dim
            # score_emb = [0.]*self.q_num
            score_emb = [[0. for col in range(1)] for row in range(self.q_num)]

            # score （q_num *1） one-hot
            if  isinstance(log['problem_id'],str):
                log['problem_id'] = eval(log['problem_id'])
                log['correct'] = eval(log['correct'])
            # for problem_id , score in zip(eval(log['problem_id']),eval(log['correct'])):
            for problem_id , score in zip(log['problem_id'],log['correct']):
                score_emb[problem_id] = [score]

            # 数据中的维度不一致
            # 1.user_k_kc 转为等长的one-hot   2.进行填充？
            user_k_kc =  [[0. for col in range(self.q_num)] for row in range( self.knowledge_num)]
            if isinstance(log['problem_id'], str):
                kn_id = eval(log['skill_id'])
            else:
                kn_id = log['skill_id']


            # for kc_list in eval(log['kn_context']):
            if isinstance(log['problem_id'], str):
                log['kn_context'] = eval(log['kn_context'])
            for kc_list,skill_id in zip( log['kn_context'],kn_id):
                # temp_emb =  [0.]*self.q_num
                for i in kc_list:
                    user_k_kc[skill_id][int(i)] = 1
                    # temp_emb[int(i)] = 1
                # user_k_kc.append(temp_emb)

            # user_k_kc = eval(log['kn_context'])
            if isinstance(log['problem_id'], str):
                ex_id = eval(log['problem_id'])
            else:
                ex_id = log['problem_id']

            stu_id = log['user_id']
            # print('debug: next_batch: stu_id:{}'.format(stu_id))



        self.ptr += self.batch_size

        return torch.LongTensor([stu_id]), torch.LongTensor(kn_id), torch.Tensor(score_emb), torch.LongTensor(user_k_kc), torch.LongTensor(ex_id)

    def get_len(self):
        # return len(self.data)
        return  self.end_num
    def is_end(self):
        # if self.ptr + self.batch_size > len(self.data):
        if self.ptr + self.batch_size >  self.end_num   :
            return True
        else:
            return False
            # return True

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, data_dim,root_path,d_type='validation'):

        self.root_path = root_path
        self.q_num = data_dim['exer_n']
        self.knowledge_num = data_dim['knowledge_n']

        self.ptr = 0
        self.data = []
        self.d_type = d_type

        if d_type == 'validation':
            data_file = root_path + '/val_data.json'
        else:
            data_file = 'data/test_set.json'
        config_file = 'config.txt'
        # data_file = root_path+'../data_set/train_data.json'
        # config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        self.q_num = data_dim['exer_n']
        self.knowledge_num = data_dim['knowledge_n']
        # self.end_num = len(self.data)
        self.end_num = 5


    def next_batch(self):
        if self.is_end():
            return None, None, None, None,None
        # print('valid data  _ len:',len(self.data))

        # print('logggggg')
        log = self.data[self.ptr]

        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        # 一个用户的一条记录'
        # print(log)

        # print('debug: next_batch() :log: \r\n{}'.format(log))
        # knowledge_emb = [0.] * self.knowledge_dim
        # score_emb = [0.]*self.q_num
        score_emb = [[0. for col in range(1)] for row in range(self.q_num)]
        # print(log)
        # score （q_num *1） one-hot
        if isinstance(log['problem_id'], str):
            log['problem_id'] = eval(log['problem_id'])
            log['correct'] = eval(log['correct'])
        # for problem_id , score in zip(eval(log['problem_id']),eval(log['correct'])):
        for problem_id, score in zip(log['problem_id'], log['correct']):
            score_emb[problem_id] = [score]

        # 1.user_k_kc 转为等长的one-hot   2.进行填充？
        user_k_kc = [[0. for col in range(self.q_num)] for row in range(self.knowledge_num)]
        if isinstance(log['problem_id'], str):
            kn_id = eval(log['skill_id'])
        else:
            kn_id = log['skill_id']

        # for kc_list in eval(log['kn_context']):
        if isinstance(log['problem_id'], str):
            log['kn_context'] = eval(log['kn_context'])
        for kc_list, skill_id in zip(log['kn_context'], kn_id):
            # temp_emb =  [0.]*self.q_num
            for i in kc_list:
                user_k_kc[skill_id][int(i)] = 1

            # user_k_kc = eval(log['kn_context'])
        if isinstance(log['problem_id'], str):
            ex_id = eval(log['problem_id'])
        else:
            ex_id = log['problem_id']

        stu_id = log['user_id']
        # print('debug: next_batch: stu_id:{}'.format(stu_id))

        self.ptr += 1

        return torch.LongTensor([stu_id]), torch.LongTensor(kn_id), torch.Tensor(score_emb), torch.LongTensor(user_k_kc), torch.LongTensor(ex_id)
    def is_end(self):
        # if self.ptr >= len(self.data):
        if self.ptr >=self.end_num:
            return True
        else:
            return False
    def get_len(self):
        return len(self.data)
    def reset(self):
        self.ptr = 0
