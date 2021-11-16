
# 添加根路径名
import os
import sys

from sklearn.metrics import roc_auc_score

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


# from distributed.protocol.tests.test_torch import torch
import torch
import torch.nn as nn
from utils.model_helper import detach_param_with_grad
from model.rbp import RBP
from memory_profiler import profile
import numpy as np



# 通过，user id 以及exer_id 等 在embedding 中选取对应的数  0
class KTModel(nn.Module):
    # @ profile(precision=4, stream=open('memory_profiler_init.log', 'w+'))
    def __init__(self, data_dim,config):
        # 根据对应id去选取
        self.knowledge_n = data_dim['knowledge_n']
        self.exer_n = data_dim['exer_n']
        self.student_n = data_dim['student_n']




        # self.stu_dim = self.knowledge_dim
        # 超参
        self.H = torch.tensor(config.model.H)
        self.threshold = float(config.model.threshold)
        self.num_update = int(config.model.num_update)
        super(KTModel, self).__init__()
        # # 掌握程度矩阵，不需要梯度传导
        # self.U_embeding =  nn.Parameter(torch.randn((self.student_n, self.knowledge_n)), requires_grad=True)
        # 没有保存用户矩阵
        # self.U_embeding = nn.Embedding(self.student_n, self.knowledge_n)

        # 有梯度的，无法更换值。
        self.U_embeding_origin =torch.rand((self.student_n, self.knowledge_n))
        #需要记录计算图，来计算 du* /du等，但是下一次eopch时，这个地方的值是没有更新的
        self.U_embeding =  nn.Parameter(self.U_embeding_origin.clone(), requires_grad=True)



        self.grad_method = config.model.grad_method
        self.truncate_iter = config.model.truncate_iter
        # self.loss_func_temp = nn.L1Loss()
        # self.loss_func_temp = My_loss()
        # self.non_linear = nn.Sigmoid()

        # 权重参数设置
        '''公式 1   a对应阿尔法，计算出的UEK是每个知识点有一个，所以是knowledge_n'''
        # zkc 是k知识点的上下文c的计算   （学生个数，知识点数量，上下文数量）
        # self.W = nn.Embedding(self.student_n,self.knowledge_n,self.exer_n - 1)
        # self.W = nn.Embedding(self.student_n,self.knowledge_n,self.exer_n )
        # 用户共享
        # ???? w >0
        # self.W = nn.Embedding(self.knowledge_n,self.exer_n )
        # 有 unsqueez操作

        # 更换参数注册方式
        # self.W = nn.Embedding(self.knowledge_n,self.exer_n )
        # 每个知识点对应一个整体的上下文（题目的数量），有可能某个知识点点不含某个上下文，所以梯度为0 设为（kn_num,1） ？??
        W =torch.rand((self.knowledge_n,self.exer_n))
        self.W = torch.nn.Parameter(W)
        self.register_parameter("W", self.W)


        # 均匀分布
        # torch.nn.init.uniform_(self.W, a=0.0, b=1.0)
        # self.W = torch.nn.init.normal_(self.W, mean=0, std=1)
        '''
        公式3
        '''
        # (kn,kc) 每个知识点的每一个上下文一个
        #
        '''
        # unsqueez操作有问题?
        '''

        # 更换参数注册方式
        beiTa_1 = torch.rand((self.knowledge_n, self.exer_n))
        self.beiTa_1 = torch.nn.Parameter(beiTa_1)
        self.register_parameter("beiTa_1", self.beiTa_1)
        # self.beiTa_1 = nn.Embedding(self.knowledge_n, self.exer_n)
        # （kc，1） 每个上下文一个
        # self.beiTa_2 = nn.Embedding(self.exer_n, 1)
        beiTa_2 = torch.rand((self.exer_n, 1))
        self.beiTa_2 = torch.nn.Parameter(beiTa_2)
        self.register_parameter("beiTa_2", self.beiTa_2)

        '''公式 5  对应ck （知识点数量+1，上下文数量）'''
        # self.B = nn.Embedding(self.knowledge_n + 1, self.exer_n -1)
        # self.B = nn.Embedding(self.knowledge_n , self.exer_n ,self.knowledge_n)
        # B =torch.rand(self.knowledge_n , self.exer_n ,self.knowledge_n,dtype=torch.float16) #cpu似乎不支持16
        B = torch.rand(self.knowledge_n , self.exer_n ,self.knowledge_n)
        self.B = nn.Parameter(B,requires_grad=True)
        '''公式7，9'''
        # 猜对的概率
        # # 失误的概率
        # 学生非共享，一个学生的一个题一个参数
        # self.guess_slip = nn.Embedding(self.student_n, self.exer_n,2)
        # 共享参数，一题一个参数， guess，slip ，
        self.guess_slip = nn.Embedding(self.exer_n,2)
        # 用户非共享参数
        # self.A = nn.Embedding(self.student_n, self.exer_n,2)
        # 用户共享参数 3对应pc的三个值
        self.A = nn.Embedding(self.exer_n,3)
        '''公式9，4'''
        # 相当于难度   初始化后可能有负数
        self.gamma_c =  nn.Embedding(self.exer_n,1)  #公式4
        self.gamma_e = nn.Embedding(self.exer_n,1)
        self.alpha = nn.Embedding(self.exer_n,1)
        self.alpha.name = 'alpha'







    # 一个知识点一个的训练
    # 使用Q，然后一个 k知识点对应的上下文（题）的矩阵应该可以一次训练出所有的uk
    # u_last:上一次的学生知识点掌握
    #kc:知识点对应得上下文，[]
    # score:上下文对应的得分 []   （ex_n, 1）
    # q_kn：每个题对应的知识点
    # kn_n: 知识点个数
    # 针对每个上下文的知识点one_hot向量
    # 计算单个知识点的
    # cur_kn :当前知识点id
    # kn_id： 知识点
    '''
    :param stu_id: 
    :param kn_id:     知识点id  
    :param q_num:     题目数量   
    :param U:         学生知识点掌握情况   
    :param score:     该学生的做题得分   1正确（q_num*1）one-hot
    :param user_k_kc: 用户在知识点k上的上下文对应（需要是做过的题）  1为有此题，0为没有
    :param d:         试题的度  （q_num,1）   
    :param q_kn:       问题对应的知识点，one——hot  
    '''
    # def forward(self, stu_id,kn_id,score,user_k_kc,ex_id,labels,q_kn,d):
    # @ profile(precision=4, stream=open('memory_profiler_forward.log', 'w+'))
    def forward(self, stu_id, kn_id, score, user_k_kc, ex_id,q_kn,d):
        # print('log:knnnn_id:{}'.format(kn_id))

        U = self.U_embeding[stu_id].T

        # nn.embeding 可以通过数组的形式，选取对应的序号值
        # 需要得数据是Q矩阵（知识点*习题），知识点对应得上下文，上下文中得知识点，以及Uk，做题记录
        #
        # 数据
        # 需要k所在上下文（试题得得分）           k ：q

        # ------------
        if self.training:

            '''
                  # 1. torch.sum(input, dim, out=None)
                  # 参数说明：
                  # 
                  # input：输入的tensor矩阵。
                  # 
                  # dim：求和的方向。若input为2维tensor矩阵，dim=0，对列求和；dim=1，对行求和。注意：输入的形状可为其他维度（3维或4维），可根据dim设定对相应的维度求和。
                 '''


            '''
            :param user_k_kc  用户在知识点k上的上下文对应（需要是做过的题）
            :param cur_kn  当前处理的知识点
            :param U_update_last  上一次的状态

            '''

            def update(self, cur_kn, U):
                '''
                WKC的计算主要由三部分组成； IC.GKC,ZC
                '''
                # 选择对应当前知识点的上下文
                # print('updatefunc cur_kn:{}'.format(cur_kn))
                # cur_kn= int(cur_kn)

                k_kc = user_k_kc[cur_kn]
                # 0，1值转成 整数形式（返回的索引,一个元组，取出其中的值）
                k_kc = torch.where(k_kc == 1)
                k_kc = k_kc[0]
                # k_kc = torch.tensor(k_kc)

                '''
                1.IC
                公式 7
                '''

                def sum_uk():
                    # 对应上下文中知识点掌握程度的和
                    # print('debug: func: U:{} \r\n u.shappe:{}'.format(U,U.shape))
                    r = torch.mm(q_kn[k_kc], U)  # (k_kc_num,kn_num) * (kn_num,0)
                    # print('debug: func: r:{} \r\n r.shappe:{}'.format(r, r.shape))
                    # r :([[1.4226],
                    #         [1.4226]]) 结果会是一样的吗？

                    return r

                dddd = d[k_kc]  #(kc_num,1)
                sss = sum_uk() #(kc_num,1)
                ssaa = score[k_kc]
                yc = (- (score[k_kc] - (sum_uk() / d[k_kc])) ** 2).exp()
                # score 为习题对应的得分

                Pc = torch.cat((1 - self.guess_slip(k_kc), yc), 1)  # (kc_n,3)
                # Pc = torch.tensor(Pc)g
                # ac = torch.rand()   #权重

                Ic = torch.sigmoid(-(torch.sum(self.A(k_kc) * Pc, dim=1)))  # 公式7  （kc,1)   A[kc] :(kc_n,3), dim=1对行求和
                Ic  = Ic.reshape(k_kc.numel(), 1)


                ''''
                2.GKC
                '''

                # threshold = 0.5 + self.threshold
                # 从Q矩阵中选取当前上下文的知识点。
                # 不共享内存
                kc_kn_gkc = torch.index_select(q_kn, 0, k_kc)# (kc_n,kn_n)    one-hot  维度需要一样
                # 去掉当前知识点,相当于置当前知识点在Q中为0
                # q_kn[k_kc, cur_kn] = 0
                kc_kn_gkc[:,cur_kn] =0
                # 需要对应多个上下文,增加维度
                U_gkc = U.repeat(1, k_kc.numel()).T  # U(kn,1) -> U_gkc(kc_n,kn_n)
                # .eq返回一个true，false矩阵。本式代表kc中不含有的知识点，其掌握程度置为阈值，代表没有贡献,以及当前知识点也要做处理
                # kc_kn_gkc
                U_gkc[kc_kn_gkc.eq(0)] = 0.5
                # 新公式6 ,(kc_n,kn_n)
                Qkc = torch.where((U_gkc > (0.5 + self.threshold)) | (U_gkc < (0.5 - self.threshold)), (U_gkc - 0.5),
                                  torch.zeros_like(U_gkc))

                # torch.nn.functional.one_hot(score, num_classes=7)
                # B :(kn_n,ex_n,kn_n)
                # 选取问题
                # 选择对应位置的权重，这种方式是否有影响？
                # B(self.knowledge_n , self.exer_n ,self.knowledge_n)

                temp = self.B[cur_kn][k_kc] * Qkc  # (kc_num,kn_num)  对应成
                # ? sum
                # 确定gkc的维度，确定是否需要sum
                Gkc = torch.sigmoid(-(torch.sum(temp, dim=1))) - 0.5  # 公式5 ，b为权重 (kc_num,1)   sum是为了 （kc_num,kn_num）-> (kc_n,1)
                Gkc = Gkc.reshape(k_kc.numel(), 1)-0.5  #(kc_num)->(kc_num,1)

                '''-----------------'''

                '''
                3.ZC
                '''
                # 公式 4   gamma 相当于困难度，de：此题的度，包含的知识点个数
                # Zc = torch.sigmoid(-self.gamma.e / de  * (score - 0.5)) -0.5
                # kc为一个知识点的上下文 q_id

                # ddd = self.gamma(k_kc) / d[k_kc]
                # sss = score[k_kc] - 0.5
                Zc = torch.sigmoid(-(self.gamma_c(k_kc) / d[k_kc]) * (score[k_kc] - 0.5))  - 0.5  # (kc_n,1)
                # torch.mul(input, other):对应乘
                # torch.mm(mat1, mat2):数学乘
                # * 对应乘：维度要一致  (kc,1) * (kc,1)

                '''
                计算WKC
                '''
                # 每个k得每个  c
                # h = self.h           #衡量上下文对知识点影响的超参数

                # 新公式2,3
                # 上下文组成了矩阵
                # h:超参，越大其影响就越大
                # WKC = h*Ic * (βkc gkc + βc Zc)
                # IC:(kc_n,1)   Gkc:(kc_n,1)   Zc:(kc_n,1)
                # 参数beita的维度都是（kn_n,ex_n）  选取当前知识点的当前上下文位置作为参数
                # 梯度更新时有没有影响？？ 一次训练多个知识点时如何处理? cur_kn =[],kc[[],[]]  for?
                # ??????beita 相乘可能有问题
                # b1 = self.beiTa_1(torch.tensor([cur_kn]))[0][k_kc].reshape(k_kc.numel(),1)   #
                # debug
                '''-------------------debug----------------------'''

                # b1 = self.beiTa_1(torch.tensor([cur_kn]))[0][k_kc].unsqueeze(dim=1)   #
                # print('b1 = {},b1.shape = {}'.format(b1,b1.shape))
                # print('b1.T = {},b1 T.shape = {}'.format(b1.T,b1.T.shape))
                # print('b1 test = {},b1 T.shape = {}'.format(torch.tensor(b1).T,torch.tensor(b1).T.shape))
                # print('Gkc = {},Gkc.shape = {}'.format(Gkc,Gkc.shape))
                # temp_2 = torch.mul(b1, Gkc)   #Gkc(1,KC_n)
                # print('temp_2 = {},temp_2.shape = {}'.format(temp_2,temp_2.shape))
                #
                # # temp_4 = torch.mul(self.beiTa_2(cur_kn, k_kc), Zc)
                # # temp_4 = torch.mul(self.beiTa_2(torch.tensor([cur_kn]))[0][k_kc], Zc)
                # b2 = self.beiTa_2(k_kc)
                # print('debug b2 = {}  b2.shape = {}'.format(b2,b2.shape))
                # print('debug Zc = {}  Zc.shape = {}'.format(Zc,Zc.shape))
                # temp_4 = torch.mul(b2, Zc)
                # print('debug temp_4 = {}  temp_4.shape = {}'.format(temp_4,temp_4.shape))
                #
                # temp_5 = torch.add(temp_2,temp_4)
                # temp_3 = torch.mul(self.H, Ic.reshape(k_kc.numel(),1))
                # print('debug temp_3 = {}  temp_3.shape = {}'.format(temp_3,temp_3.shape))
                #
                # WK =torch.mul( temp_3 , temp_5)# (kc_n,1)
                # print('debug WK = {}  WK.shape = {}'.format(WK,WK.shape))
                '''-------------------debug----------------------'''
                # 选取应该有问题
                # b1 = self.beiTa_1(torch.tensor( [cur_kn]))[0][k_kc].unsqueeze(dim=1)  # （kc_num，1）
                # b2 = self.beiTa_2(k_kc)  # （kc_num，1）

                # WK =torch.mul( torch.mul(self.H,Ic) ,(torch.mul(b1,Gkc) + torch.mul(self.beiTa_2(cur_kn,k_kc) , Zc) ))# (kc_n,1)
                # reshape？
                # WK = torch.mul(torch.mul(self.H, Ic ),
                #                torch.add(torch.mul(b1, Gkc), torch.mul(b2, Zc)) )  # (kc_n,1)
                # WK = torch.mul(torch.mul(self.H, Ic ),
                #                torch.add(torch.mul(self.beiTa_1(torch.tensor( [cur_kn]))[0][k_kc].unsqueeze(dim=1), Gkc), torch.mul(self.beiTa_2(k_kc), Zc)) )  # (kc_n,1)
                # dd = self.beiTa_1[cur_kn][k_kc].unsqueeze(dim=1)
                # ddd = self.beiTa_2[k_kc]
                # mm =  torch.mul(self.H, Ic)
                # m1= torch.mul(dd, Gkc)
                # m2 =  torch.mul(self.beiTa_2[k_kc], Zc)


                WK = torch.mul(torch.mul(self.H, Ic),
                               torch.add(torch.mul(self.beiTa_1[cur_kn][k_kc].unsqueeze(dim=1), Gkc),
                                         torch.mul(self.beiTa_2[k_kc], Zc)))
                # mul对应相乘
                # ψkc = λIc (βkc gkc + βc Zc)

                '''
                计算 最后的学生知识点掌握
                '''

                # 学生未共享权值
                # u_k_new = torch.sigmoid(-(self. W[stu_id][cur_kn][kc].reshape(kc.numel(),1) * WK));  #公式1
                # 学生共享参数  WK(KC_N,1)   W[cur_kn][kc]:(1，kc_n)   注意：cur_kn必须为单值。
                # 更新知识点掌握程度
                # cur_kn 为单值 否则取值时会有问题
                # u_k_new = torch.sigmoid(-(tor
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                # ch.mm(self. W(cur_kn)[k_kc].unsqueeze(dim=0) , WK)));  #公式1(1,1)
                # 待测      unsqueeze(dim=0) 在维度为0的地方加上 一维
                # u_k_new = torch.sigmoid(-(torch.mm(self.W(cur_kn)[k_kc].unsqueeze(dim=0), WK)));  # 公式1(1,1)
                # WK
                # wwww = self.W[cur_kn][k_kc].unsqueeze(dim=0)
                # 维度不对
                u_k_new = torch.sigmoid(-(torch.mm(self.W[cur_kn][k_kc].unsqueeze(dim=0),WK))) # 公式1(1,1)  (1,kc_num) * (kc_num,1)

                # debug
                # wc = self.W(cur_kn)
                # print('debug: wc : {}'.format(wc))
                # wc = wc[k_kc].unsqueeze(dim=0)
                # print('debug: wc2 : {}'.format(wc))
                # torch.mm(wc, WK)
                # u_k_new = torch.sigmoid(-(torch.mm(wc,WK)));  #公式1  (1,1)
                # debug

                diff = u_k_new - U[cur_kn]
                # print('知识点：{} , 掌握状态 old：{},new :{},diff:{}' .format(cur_kn,U[cur_kn] ,u_k_new,diff))
                return u_k_new

            '''
                      for k  in  kn
                           for num in num_update
                               update Uk 
                     计算predict
                     累加loss
               '''
            #         update
            #        构建 (num_update+1,kn_n+1)的数组      (2009 知识点id从1开始)   最后一个存储初始u   应该不用none
            U_update = [[None for col in range(self.knowledge_n + 1)] for row in range(self.num_update + 1)]
            # diff_norm =[[None] * self.knowledge_n+1  for i in range(0,self.num_update)]
            diff_norm = [[None for col in range(self.knowledge_n + 1)] for row in range(self.num_update)]
            # 最后一个设为 初始的状态
            U_update[-1] = U
            # print("nummmm:",self.num_update)
            for num in range(self.num_update):
                # print('状态更新次数：', num)

                # 有些知识点并不会计算到,所以等于上一个的值
                # 参加不参加梯度传导？应该参加  怎么记录  list里 tensor 还是整个tensor
                # clone对梯度传播的影响?
                U_update[num] = U_update[num - 1].clone()
                # 这种复制方法 可能会让计算图混乱   或者detach后放进去
                # ten =  torch.empty((1,self.knowledge_n + 1))
                # ten =   U_update[num - 1].clone()
                for cur_kn in kn_id:
                    # print('处理知识点：', cur_kn)
                    # print('cur_id: {} \r\n kn_id:{}'.format(cur_kn,kn_id))
                    # update一次处理所有的，然后返回一个tensor就行了
                    # ten[0][cur_kn] = _update(self, cur_kn, U_update[num - 1])
                    # 几次之后知识点的变动就很小了
                    U_update[num][cur_kn] = update(self, cur_kn, U_update[num - 1])
                # U_update[num] = ten.clone()
                # 3-4次之后就不变了？   按道理 所有的知识点都应该计算过了
                # UUU = torch.norm(U_update[num] - U_update[num - 1])
                # test = UUU.item()
                diff_norm[num] = torch.norm(U_update[num] - U_update[num - 1]).item()


                # ????????????
                # if self.grad_method == 'TBPTT':
                #     if num + 1 + self.truncate_iter == self.num_update:
                #         U_update[num] = U_update[num].detach()
                # elif 'RBP' in self.grad_method:
                #     if num + 2 == self.num_update:
                #         # u上应该没有梯度？？？   这里detach 是为了重新进入一个计算图？  检查此处   为什么 倒数第二个剥离？与本论文又有什么不同
                #         # clone的影响
                #         # 为什么要detach  本质是最后一个状态对这个值的求导，论文里则是最后一个状态对第一个值的求导，所以要对第一个值做detach？为什么detach？ u在梯度传播后是否会更新  debug
                #         U_update[num] = detach_param_with_grad(U_update[num])
                #         # U_update[num] = detach_param_with_grad((U_update[num]))[0]

            # 替换为新的数据
            # 让它有梯度？
            # self.U_embeding[stu_id] = torch.tensor(U_update[-2]).T.clone().detach()
            self.U_embeding_origin[stu_id] = torch.tensor(U_update[-2]).T.clone().detach()

            state_last = U_update[-2]
            # state_2nd_last = U_update[-3]
            # U_update[-1].requires_grad = True
            # 倒数第二个状态
            state_2nd_last = U_update[-3]
            # state_1nd = U_update[-1]

            # supervised_process
            '''--------supervised_process---------'''
            # predict = supervised_process(self,q_kn,ex_id,state_last,d)
            Ukse = torch.div(torch.mm(q_kn[ex_id], state_last), d[ex_id]) - 0.5  # (ex_choice_num,1)
            # score 对应的论文里的v0
            predict = torch.sigmoid( -(torch.mul(self.alpha(ex_id), Ukse) + self.gamma_e(ex_id)))  # (ex_choice_num,1)
# chnge            # 计算当前损失
            # loss,v1,v2,v3 = self. loss_func_temp(score,predict,state_last,ex_id,q_kn)
            # line10 的  l/u*
            # grad_state_last = torch.autograd.grad(loss, state_last, retain_graph=True,create_graph = True)
            # 这里是否包括U，u是不用梯度更新的
            # ppp有问题
            # params = [pp for pp in self.parameters()]
            # print('参数 查看')

            # print('当前损失值：{} predict：{}'.format(loss, predict))
            # 大于0.5的算作1 正确
            # 预测的正确率
            # predict[predict.ge(0.5)] = 1
            # predict[predict.lt   (0.5)] = 0
            # l = predict - score[ex_id]
            # accuracy  = torch.sum(l == 0) / l.numel()


            # print('当前预测正确率：{}'.format(torch.sum(l == 0) / l.numel()))



         # # 将loss以及rbp的梯度计算移到外面去
         #
         #    if 'RBP' in self.grad_method:
         #        # line 12
         #       change
         #         grad = RBP(params,
         #             #       求导时，有中间循环的过程吗
         #             # 有：需要考虑中间的状态值，copy？怎么只是复制其值，导数会传 过来  ，grad的问题
         #             # 没有：怎么将两个状态连起来      rbp里面时最后两个值的导数
         #                   [state_last],
         #                   [state_2nd_last],
         #                   grad_state_last,
         #                   # update_forward_diff=_update_forward_diff,
         #                   update_forward_diff  =None,
         #                   # 论文中的 line 11， M
         #                   truncate_iter=self.truncate_iter,
         #                   rbp_method=self.grad_method)
         #    else:
         #         print()
         #        # params = [pp for pp in self.parameters()]
         #        # grad = torch.autograd.grad(loss, params)
        else:
            # 验证
            # print('验证')
            # predict = supervised_process(self,q_kn, ex_id, U, d)
            Ukse = torch.div(torch.mm(q_kn[ex_id], U), d[ex_id]) - 0.5  # (ex_choice_num,1)
            # score 对应的论文里的v0
            predict = torch.sigmoid(-(torch.mul(self.alpha(ex_id), Ukse) + self.gamma_e(ex_id)))
            state_last = U
            # loss,v1,v2,v3 = self.loss_func_temp(score, predict, U, ex_id, q_kn)
            state_2nd_last = None
            grad = None
            diff_norm = None




        # return loss,v1,v2,v3, state_last, diff_norm, grad,accuracy,predict
        # return loss,v1,v2,v3, state_last, diff_norm, grad,predict
        return   state_last, state_2nd_last,predict,diff_norm


#
 # 对某一道题的预测，根据这个题的知识点掌握程度
# input：问题矩阵（知识点表示）， 学生知识点掌握i程度  sum()/度  -0.5
# 输出：是否作对
# ?/几道题的预测
# @ profile(precision=4, stream=open('memory_profiler_supervised_process.log', 'w+'))
def  supervised_process(self,q_kn,ex_id,U,d):
    Ukse = torch.div(torch.mm(q_kn[ex_id],U) , d[ex_id]) - 0.5  #(ex_choice_num,1)
    # score 对应的论文里的v0
    score_predict = torch.sigmoid(-(torch.mul(self.alpha(ex_id),Ukse) + self.gamma(ex_id)))  #(ex_choice_num,1)
    return score_predict



# 测试一下
# loss 2个 各乘权值（总为1）
# 平均值+-方差 稳定性
class My_loss(nn.Module):
     def __init__(self):
         super().__init__()

     # @profile(precision=4, stream=open('memory_profiler_LOSS_mean_min_process.log', 'w+'))
     def mean_min_process(self,target,ex_id,q_kn,U):
         # s_right = target.eq(0)
         # s_wrong = target.eq(1)
         s_right = target.eq(1)
         s_wrong = target.eq(0)
         # print('s_right:{} \r\n s_wrong:{}'.format(s_right, s_wrong.squeeze()))
         # 选择正确和错误的exid,可能为空
         ex_1 = ex_id[s_right.squeeze()]
         ex_0 = ex_id[s_wrong.squeeze()]
         mean_u, min_u = torch.tensor([]),torch.tensor([])
         # print('U:{}'.format(U))  #torch.Size([379, 1])
         # U = U.squeeze()   #[379]   mm(KN_1,U.T)  or   mul(kn_1,U，ex_num) .mean(dim =1)
         # U.repeat(1, ex_1.numel())
         if min(ex_0.shape)  != 0:

             # 选取习题包含的知识点
             kn_0 = q_kn[ex_0]  # (5,379)  [[],[],[]]
             # 选取对应的
             temp_0 = U.repeat(1, ex_0.numel()).T * kn_0
             # 等于0的位置置为2 相当于最大值
             one = torch.ones_like(temp_0)
             temp_0 = torch.where(temp_0 == 0, one, temp_0)
             min_u = temp_0.min(dim=1)[0]

         # 找到每道题的知识点
         if min(ex_1.shape) != 0:

             kn_1 = q_kn[ex_1]    #(8,379)   （ex_num,kn_n）
             # repeat调整U为 (ex_num,kn_n)
             # 对行求均值
             U_1 = U.repeat(1, ex_1.numel()).T#torch.Size([379, 8])
             temp_1 =  U_1 * kn_1
             # kn_1的1的个数，相当于统计包含知识点个数
             kn_count = kn_1.sum(dim = 1)
             temp_2 = temp_1.sum(dim =1)
             mean_u = temp_2/kn_count

         # 对行取最小值

         # U -> 与kn_1一致的维度
         # mean_u = mean_u.detach()
         # min_u = min_u.detach()
         return mean_u,min_u

     # @profile(precision=4, stream=open('memory_profiler_LOSS_forward.log', 'w+'))
     def forward(self, target, predict,U,ex_id,q_kn):
         # 一个题对应一个均值，和最小值
         target = target[ex_id]
         index_1 = torch.where(target == 1)[0]
         index_0 = torch.where(target == 0)[0]

         mean_u,min_u = self.mean_min_process(target,ex_id,q_kn,U)
         if min(mean_u.shape) != 0:
             # d = predict[index_1].squeeze()
             # mm = mean_u
             # sss = torch.sub(d, mean_u)
             # temp_2 = torch.sum(torch.norm(torch.sub(predict[index_1].squeeze(), mean_u)))
             t = torch.sub(predict[index_1].squeeze(), mean_u)
             temp_2 = torch.norm(torch.sub(predict[index_1].squeeze(), mean_u))

         else:
             temp_2 = torch.tensor(0)

         if min(min_u.shape) != 0:
             t =torch.sub(min_u, predict[index_0].squeeze())
             temp_3 = torch.norm(torch.sub(min_u, predict[index_0].squeeze()))

             # ddd = torch.norm(torch.sub(min_u, predict[index_0].squeeze()))
             # temp_3 = torch.sum(torch.norm(torch.sub(min_u, predict[index_0].squeeze())))
         else:
             temp_3 = torch.tensor(0)

         # 原始
         # result = torch.sum(torch.norm(torch.sub(target,predict),dim=1))+ torch.sum(torch.norm(torch.sub(predict[index_1],mean_u),dim=1))+ torch.sum(torch.norm(torch.sub(min_u,predict[index_0]),dim=1))
         # result = torch.sum(torch.norm(torch.sub(target,predict),dim=1))+ torch.sum(temp_1,dim=1)+ torch.sum(temp_2,dim=1)
         # ssub = torch.sub(target, predict)
         # noo = torch.norm(torch.sub(target, predict))
         temp_1 = torch.sum(torch.norm(torch.sub(target, predict)))
         result = temp_1 + temp_2 + temp_3
         # print('debug result:{} \r\n result :{}'.format(result, result))
         # ?有没有高效保存的方式  线程？  顺序呢
         # with open('result/loss_val.txt', 'a', encoding='utf8') as f:
         #     # f.write('式1= %f,  式2= %f, 式3 = %f ， sum = %f\n' % (v1, temp_1, temp_2,result))
         #     f.write(' %f, %f,  %f ， %f,%s\n' % (v1, temp_1, temp_2,result,str(predict.tolist())))


         #     记录predict



         return result,temp_1,temp_2,temp_3
'''
tensor求导
https://blog.csdn.net/weixin_43012796/article/details/108294620
toch.norm 求范数
https://blog.csdn.net/goodxin_ie/article/details/84657975
#         计算一个学生在知识点k上的掌握程度    （单个，还是能以矩阵形式，一次计算处所有的知识点掌握）
#               权重 * Uek ：    Uexk：与知识点k相关的习题    公式 2
 #                    （1*1）Uek:  累加：与k知识点相关的习题的得分 （-0.5？）  / 与知识点k相关的试题的数量
#                               习题得分是 0或1的情况下，减去0.5的意义
#               权重矩阵 * Zk
#               Zk =^ Zkc  衡量上下文知识点对k的影响， 由Zkc组成
                        找到Ic和Gkc的维度，Zkc应该为1*1
                        Zkc: h（Ic* Gkc - 0.5）   公式（4）   h为超参数      (2*kn) =   (2*1) * (1*kn)??? 衡量c对k的影响
                            Gkc： sigmoid（- （Bkc * Ck））   b：权重  （Kn*1） 公式 5 如权重是（1*kn）？ 
                                Ck：Qc||Qkc     (kn*1)?
                                     Qc = Ve - 0.5 /De     De为这道题含有的知识点个数，衡量做题情况对知识点掌握的影响
                                        (1*1)？ 每次计算时是以k为点，如果上下文中有多个题，此时这里还是1*1？？？
                                     Qkc =^ Lki     i属于C，知识点k的上下文   （kn-1 * 1） ?
                                         Lki =    U'ki - 0.5  , U'ki 不属于 0.5 +- v      （ 1*1 ） 公式 6  
                                             0          , U'ki 属于 0.5 +- v
                            Ic = sigmoid（-（Ac * Pc） Ac权重    （kn-1 * 2） ?    （1*2） =（1*1） * （1*2） ？ 
                                Pc =^  (1-P(Sc) , 1- P(Gc))    P(Sc) : (Kn-1 * 1)?(1 * kn -1) or(1*1) ?
                                                               p(Gc) : (Kn-1*1). ? (1* kn -1)
                                                               Pc    : (Kn-1 * 2) ? (2* kn-1)  or (1*2)


'''


