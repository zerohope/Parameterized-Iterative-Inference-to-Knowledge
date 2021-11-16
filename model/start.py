import KTModelRunner_2
import KTModelRunner_newstu
from Constant2 import Constants as config
import time
if __name__ == '__main__':
    start = time.time()
    # runner = KTModelRunner.KTModelRunner(config)
    # 超参影响
    dataset = ['math1','math2','assist2015']
    type = ['epsilon','lambda']
    type = ['lambda']
    epsilon = [0.5,0.1,0.01,0.001]
    lambd = [20,15,10 ,5,1,0.5 , 0.1 ,0.001]
    # lambd = [10 ,15,20]
    epsilon_v = 0.001
    lambd_v = 0.1
    for hp in type:
        config.model.hp_type = hp
        if hp == 'epsilon':
            print('epsilon')
            config.model.H = lambd_v
            for d_name in dataset:
                config.DATASET = d_name
                for v in epsilon:
                    config.model.threshold = v
                    # 所有学生的训练
                    runner = KTModelRunner_2.KTModelRunner(config)
                    # runner = KTModelRunner_newstu.KTModelRunner(config)
                    runner.train()
                    end = time.time()
                    print('花费时间：{}'.format(end - start) )
        else:
            config.model.threshold = epsilon_v
            for d_name in dataset:
                config.DATASET = d_name
                for v in lambd:
                    config.model.H = v
                    runner = KTModelRunner_2.KTModelRunner(config)
                    # runner = KTModelRunner_newstu.KTModelRunner(config)
                    runner.train()
                    end = time.time()
                    print('花费时间：{}'.format(end - start))



    # runner = KTModelRunner_2.KTModelRunner(config)
    # # runner = KTModelRunner_newstu.KTModelRunner(config)
    # runner.train()
    # end = time.time()
    # print('花费时间：{}'.format(end - start))

    # 1 epoch  960s
