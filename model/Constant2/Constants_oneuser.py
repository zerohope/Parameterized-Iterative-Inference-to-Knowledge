Dpath = '../../KTDataset'

datasets = {
    'assist2009' : 'assist2009',
    'assist2015' : 'assist2015',
    'assist2017' : 'assist2017',
    'static2011' : 'static2011',
    'kddcup2010' : 'kddcup2010',
    'synthetic' : 'synthetic',
    'math1':'math1'
}

# question number of each dataset
# numbers = {
#     'assist2009' : 124,    ?????????????????筛选了选择题？？？？？？
#     'assist2015' : 100,
#     'assist2017' : 102,
#     'static2011' : 1224,
#     'kddcup2010' : 661,
#     'synthetic' : 50
# }
# numbers = {
#     'assist2009' : 124,
#     'assist2015' : 100,
#     'assist2017' : 102,
#     'static2011' : 1224,
#     'kddcup2010' : 661,
#     'synthetic' : 50
# }
# 训练时选取的数据集  用来选取维度
# DATASET = datasets['static2011']
# DATASET = datasets['assist2009']
DATASET = datasets['math1']
# NUM_OF_QUESTIONS = numbers['static2011']
# the max step of RNN model
# MAX_STEP = 50
# BATCH_SIZE = 64
# LR = 0.002
# EPOCH = 1000
# #input dimension
# INPUT = NUM_OF_QUESTIONS * 2
# # embedding dimension
# EMBED = NUM_OF_QUESTIONS
# # hidden layer dimension
# HIDDEN = 200
# # nums of hidden layers
# LAYERS = 1
# # output dimension
# OUTPUT = NUM_OF_QUESTIONS

data_dim = {
'assist2009' : {
    'student_n' : 4162,
    'exer_n' : 17751,
    # 实际值为123，这里未作映射
    'knowledge_n' : 379,
},
    'math1':{
        'student_n': 4209,
        'exer_n': 20,

        'knowledge_n': 11,
    }
}
use_gpu = False
# use_gpu = True
gpus = [0]
seed =  1234
save_dir='save'
# linux
# save_dir='/home/server/kt_model/save'
# save_dir='/home/server/kt_model/save'
# model = {
#   'name': 'HopfieldNet',
#   'input_dim': 784,
#   'hidden_dim': 1024,
#   #   状态稳定更新次数
#   'num_update': 50,
#   'grad_method': 'Neumann_RBP',
#   # grad_method: CG_RBP
#   # grad_method: RBP
#   # grad_method: TBPTT
#   # grad_method: BPTT
#   #   对应论文算法 line11 的更新次数
#   'truncate_iter': 20}

class model :
    # name = 'HopfieldNet'
    # input_dim = 784
    # hidden_dim = 1024
    #   状态稳定更新次数
    # num_update = 50
    num_update = 20

    grad_method= 'Neumann_RBP'
    # grad_method= CG_RBP
    # grad_method= RBP
    # grad_method= TBPTT
    # grad_method = BPTT
    #   对应论文算法 line11 的更新次数
    # truncate_iter = 20
    truncate_iter = 20
    #对应公式3的 λ
    # 看一致性
    H = 0.7
    # 对应公式 6的  e   掌握程度对表现影响的阈值。  属于 0.5 +- e的是无影响的。？？ 0.5的是掌握又没掌握的?
    threshold = 0.4

class train:
    # optimizer = 'Adam'
    optimizer = 'SGD'
    lr_decay = 0.1
    # lr_decay_steps = [10000]
    lr_decay_steps = [3]
    num_workers = 0
    max_epoch =100
    batch_size = 1
    display_iter = 1
    snapshot_epoch = 100000
    valid_epoch = 10000
    lr = 0.8
    wd = 0.1

    momentum = 0.9
    # lr = 0.8
    # momentum = 0.5
    # 0.625
    #
    # lr = 0.85
    # momentum = 0.4
    #  0.875

    # 循环更新的u状态，在2到3次之后就会为0？？？？  但update大的话，loss确实会偏小。
    # v1 v2（最少会到0.几） v3会逐步下降，且准确率高的话，值都会较小 。  并且在v1 v2继续下降的情况下，准确率可能并不发生变化？ 掌握程度对于做题正确的推动没有到达一个点上？（掌握程度分层？）
    # Acurracy @ epoch 0076 iteration 00000002 = 0.75
    # 训练结果： {'epoch': 76, 'data': [{'train_loss': 4.26686954498291, 'v1': 1.6279956102371216, 'v2': 0.9912902116775513, 'v3': 1.6475838422775269, 'train_accuracy': 0.75, 'iter_count': 1, 'diff_norm': '[array(0.88494974, dtype=float32), array(5.9604645e-08, dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32)]'}]}
   #Loss @ epoch 0021 iteration 00000002 = 6.991859436035156
#Acurracy @ epoch 0021 iteration 00000002 = 0.25
#训练结果： {'epoch': 21, 'data': [{'train_loss': 6.991859436035156, 'v1': 2.301438331604004, 'v2': 1.218991994857788, 'v3': 3.4714293479919434, 'train_accuracy': 0.25, 'iter_count': 1, 'diff_norm': '[array(0.5695731, dtype=float32), array(1.8235223e-05, dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32), array(0., dtype=float32)]'}]}

    momentum = 0.4
    # shuffle = True
    is_resume = False
    resume_model = None

# train = {
#   'optimizer': 'Adam',
#   'lr_decay': 0.1,
#   'lr_decay_steps': [10000],
#   'num_workers': 0,
#   'max_epoch': 1000,
#   'batch_size': 1,
#   'display_iter': 10,
#   'snapshot_epoch': 1000,
#   'valid_epoch': 10,
#   'lr': 1.0e-3,
#   'wd': 0.0,
#   'momentum': 0.9,
#   'shuffle': True,
#   'is_resume': False,
#   'resume_model': None
# }
