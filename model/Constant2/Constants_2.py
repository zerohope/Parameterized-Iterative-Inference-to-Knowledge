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
# save_dir='save'
# linux
save_dir='/home/server/kt_model/save'
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
    num_update = 50
    grad_method= 'Neumann_RBP'
    # grad_method= CG_RBP
    # grad_method= RBP
    # grad_method= TBPTT
    # grad_method = BPTT
    #   对应论文算法 line11 的更新次数
    # truncate_iter = 20
    truncate_iter = 10
    #对应公式3的 λ
    H = 0.1
    # 对应公式 6的  e
    threshold = 0.1

class train:
    optimizer = 'Adam'
    lr_decay = 0.1
    lr_decay_steps = [10000]
    num_workers = 0
    max_epoch = 5
    batch_size = 1
    display_iter = 10
    snapshot_epoch = 1
    valid_epoch = 10
    lr = 1.0e-3
    wd = 0.0
    momentum = 0.9
    shuffle = True
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
