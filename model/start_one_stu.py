import KTModelRunner_2
import KTModelRunner_newstu
import KTModelRunner_one_stu
from Constant2 import Constants as config
import time
if __name__ == '__main__':
    start = time.time()


    # runner = KTModelRunner_2.KTModelRunner(config)
    # runner = KTModelRunner_allstu.KTModelRunner(config)
    runner = KTModelRunner_one_stu.KTModelRunner(config)
    # runner = KTModelRunner_newstu.KTModelRunner(config)
    # runner = KTModelRunner_newstu.KTModelRunner(config)
    runner.train()
    end = time.time()
    print('花费时间：{}'.format(end - start))

    # 1 epoch  960s
