import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net1(torch.nn.Module):
    def __init__(self,ex_num,kn_num,U_hidden_layer,h,thereshhold):
        super(Net1, self).__init__()
        self.h = h
        self.thereshold = thereshhold
        # 上下文的最大数量对应于习题数量，为了区分起了不同的名字
        kc_num = ex_num
        self.activate = nn.Sigmoid()
        self.U_layer = nn.Linear(kc_num,U_hidden_layer)
        self.Z_layer = nn.Linear(ex_num,ex_num)

        # 三维
        self.G_layer = nn.Linear()

        self.I_layer = nn.Linear(ex_num,3)


        # 用来计算公式3
        # L1的权值应该为两维的
        self.L1 = nn.Linear(kc_num,kc_num)
        self.L2 = nn.Linear(kc_num,kc_num)







        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv(x)), 2)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.dense1(x))
        # x = self.dense2(x
        # 获取训练中的一些数据
        get_data()
        # 获取当前知识点的上下文
        k_kc = user_k_kc[cur_kn]
        # 0，1值转成 整数形式（返回的索引,一个元组，取出其中的值）
        k_kc = torch.where(k_kc == 1)
        k_kc = k_kc[0]

        # 设置z的输入 维度要一致，没有涉及的kc？
        # 输入数据返回一个新的embedding？这是一次update。更新在函数内部进行
        (score[k_kc] - 0.5)/d[k_kc]
        k_kc = user_k_kc[cur_kn]


        # 0，1值转成 整数形式（返回的索引,一个元组，取出其中的值）
        k_kc = torch.where(k_kc == 1)
        k_kc = k_kc[0]

        Z = self.Z_layer()
        Z = Z - 0.5

        # 设置G的输入
        G = self.G_layer()
        G = G - 0.5

        # 设置I的输入
        I = self.I_layer()

        # 公式3 的计算
        T = self.h * I * (self.L1(G) + self.L2(Z))

        u = self.U_layer(T)



