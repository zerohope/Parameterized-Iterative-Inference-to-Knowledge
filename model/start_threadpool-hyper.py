import threading

from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED, ALL_COMPLETED


import KTModelRunner_2
import logging as logger
from Constant2 import Constants as config
import time
import multiprocessing
from data_loader import  TrainDataLoader
def thread_pool_callback(worker):
    logger.info("called thread pool executor callback function")
    worker_exception = worker.exception()
    if worker_exception:
        logger.exception("Worker return exception: {}".format(worker_exception))


def worker(q):
    runner = q.get()
    runner.train()
if __name__ == '__main__':
    start_time = time.time()
    # runner = KTModelRunner.KTModelRunner(config)

    print("CPU的核数为：{}".format(multiprocessing.cpu_count()))
    cpu_num = multiprocessing.cpu_count()
    # 获取总数据长度
    root_path = '../data_set/' +  config.DATASET
    data_dim =  config.data_dim[ config.DATASET]


    data_loader = TrainDataLoader(data_dim,root_path,0,0)
    len = data_loader.get_all_len()
    print(len)
    # 向上取整
    batch = len//(cpu_num-2)
    # batch = 5
    c = int(cpu_num / 2)
    end = 0
    pool =  ThreadPoolExecutor(cpu_num-1)
    task_list = []
    # 超参
    str_1 = ['10,0.001', '10,0.01', '10,0.1', '10,0.5']
    str_2 = ['10,0.001', '5,0.001', '1,0.001', '0.5,0.001', '0.001,0.001']
    # 修改为每个进程运行的超参数不同
    for item in str_1:
    #     测试修改每个config后，在每个进程中是否是一样的
        hp = item.split(',')
        config.model.H = float(hp[0])
        config.model.threshold  = float(hp[1])
        start = 0
        end = 2
        print('start and end', start, end)
        runner = KTModelRunner_2.KTModelRunner(config, start, end)
        # runner = KTModelRunner_2.KTModelRunner(config,0,1)
        time.sleep(2)
        thread = threading.Thread(target=runner.train)
        # thread.start()
        task = pool.submit(runner.train)
        task.add_done_callback(thread_pool_callback)
        task_list.append(task)


    #
    # for i in range(cpu_num -2):
    # # for i in range(2):
    #     # 需要指明每一个进程需要处理哪些数据，实际上是传给dataloader
    #     # start,end    is_end: < end
    #     start = end
    #     end = (i+1) * batch
    #     print('start and end',start,end)
    #
    #     runner = KTModelRunner_2.KTModelRunner(config,start,end)
    #     # runner = KTModelRunner_2.KTModelRunner(config,0,1)
    #     thread = threading.Thread(target=runner.train)
    #     # thread.start()
    #     task = pool.submit(runner.train)
    #     task.add_done_callback(thread_pool_callback)
    #     task_list.append(task)
    # # 处理剩下的数据
    # if( end < len ):
    #     start = end
    #     end = len
    #     runner = KTModelRunner_2.KTModelRunner(config, start, end)
    #     task = pool.submit(runner.train)
    #     task_list.append(task)


    # pool.join()
    result=[]

    # wait(task_list, return_when=FIRST_COMPLETED)
    wait(task_list, return_when=ALL_COMPLETED)
    print('wait。。。。。')
    for future in as_completed(task_list):
        future
        r = future.result()
        result.append(r)
        print('结果',r)


    # 计算整体的平均值
    #  res = {'train_loss_mean':train_loss_mean_stu_all/data_len,'v1_mean':v1_mean_stu_all / data_len,'v2_mean':v2_mean_stu_all / data_len, 'v3_mean':v3_mean_stu_all / data_len,'accuracy':accuracy_mean_stu_all / data_len,
    #            'rmse_mean':rmse_all/data_len,'auc_mean':auc_all/data_len}
    # result.sum/len(task_list)
    for r in result:
        print('返回结果类型：',type(r))
        print(r)
    import  pandas as pd
    # 使用pandas 处理所有进程中返回的数据
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    data = pd.DataFrame(result)
    print(data)
    data = data.mean().to_frame()
    lam = config.model.H
    epsinol = config.model.threshold
    # 先保存各个进程返回的数据在做处理
    data.to_csv('thread_result/hyperparameters_test/'+ config.DATASET +'lam,epsinol = ' +str(lam) + ','+ str(epsinol) +'_tmp_thread_result.csv',index = False)
    # 求各列的平均值

    end_time = time.time()
    print('花费时间：{}'.format(end_time - start_time))