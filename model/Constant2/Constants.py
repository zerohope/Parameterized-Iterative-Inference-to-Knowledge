Dpath = '../../KTDataset'

datasets = {
    'assist2009' : 'assist2009',
    'assist2015' : 'assist2015',
    'assist2017' : 'assist2017',
    'static2011' : 'static2011',
    'kddcup2010' : 'kddcup2010',
    'synthetic' : 'synthetic',
    'math1':'math1',
    'math2':'math2',
}

# 训练时选取的数据集  用来选取维度
# DATASET = datasets['static2011']
# DATASET = datasets['assist2009']
# DATASET = datasets['math1']
# DATASET = datasets['math2']
# DATASET = datasets['math1']
DATASET = datasets['assist2015']

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
    },
    'assist2015': {
        'student_n': 1965,  #log15:1965     all:19917
        'exer_n': 100,

        'knowledge_n': 100,
    },
    'math2': {
        'student_n': 3911,
        'exer_n': 20,

        'knowledge_n': 16,
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


class model :

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
    #对应公式3的 λ.控制其它知识点贡献多少的参数     相当于由其它知识点传递的信息，来生成新的知识点掌握程度
    # 看一致性
    # H =0.1
    H =10
    # 对应公式 6的  e
    # threshold = 0.001
    # threshold = 0.01
    threshold = 0.1
    # threshold = 0.01
    hp_type = 'epsilon'
    # hp_type = 'lambda'

class train:
    optimizer = 'Adam'
    # optimizer = 'SGD'
    lr_decay = 0.2
    lr_decay_steps = [10]
    num_workers = 0
    max_epoch =20
    batch_size = 1
    display_iter = 1
    snapshot_epoch = 100000
    valid_epoch = 5
    lr = 0.8
    wd = 0.4

    momentum = 0.001
    # lr = 0.8
    # momentum = 0.5
    # 0.625
    #
    # lr = 0.85
    # momentum = 0.4
    #  0.875


    # momentum = 0.4

    is_resume = False
    resume_model = None


