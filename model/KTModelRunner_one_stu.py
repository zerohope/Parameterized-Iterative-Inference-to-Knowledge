from __future__ import (division, print_function)

import json
import os
import pickle
import threading

import numpy as np
import pandas as pd
# 添加根路径名
import sys
import os

from rbp import RBP

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
from torchvision import datasets
# from model import *
# from dataset import *
from collections import defaultdict

# from utils.arg_helper import mkdir
# from utils import get_logger
from utils.train_helper import snapshot, load_model
# from Constant2 import Constants as C
from KTModel import KTModel, My_loss
from data_loader_onestu import TrainDataLoader, ValTestDataLoader
from sklearn.metrics import roc_auc_score
import gc
# logger = get_logger('exp_logger')

# __all__ = ['HopfieldRunner']

#

def binarize_data(data):
  data[data > 0.0] = 1.0
  data[data <= 0.0] = 0.0




# 此版本 一个模型是一个人的数据



class KTModelRunner(object):

  def __init__(self, config):
    self.config = config
    # self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train    
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    # mkdir(self.dataset_conf.path)

  def train(self):
    # print('process:{}begin with  data start:{} end:{} '.format(threading.current_thread().name,self.start,self.end))
    print('config test h ={},epsilon={}'.format(self.config.model.H,self.config.model.threshold))

    data_dim = self.config.data_dim[self.config.DATASET]
    # model = KTModel(self.config.data_dim[self.config.DATASET],self.config)
    loss_func_temp = My_loss()

    # 模型的w参数初始化为 >0，W为rand初始化时，值在0到1之间
    # for m in model.modules():
    #   if isinstance(m,KTModel):
    #     torch.nn.init.uniform_(m.W.weight, a=0.0, b=1.0)
    #     # print('after:{}'.format(m.W.weight ))
    #     break

    # 传递文件位置
    # root_path = 'D:/文档\知识追踪工程/teacherModel/kt_model/data_set'
    # root_path = 'D:/文档\知识追踪工程/teacherModel/kt_model/data_set/' +self.config.DATASET
    root_path = '../data_set/' +self.config.DATASET
    # linux使用
    # root_path = '/home/server/kt_model/data_set/' + self.config.DATASET
    train_data_loader = TrainDataLoader(data_dim, root_path)
    val_data_loader = ValTestDataLoader(data_dim, root_path)





    # Training Loop

    # results = defaultdict(list)
    # common data
    q_kn,d = train_data_loader.getCommonData()
    display_accuracy = 0
    accuracy_all = 0
    # 用来计算每一个epoch的平均值
    # train_loss_mean_all,v1_mean_all,v2_mean_all,v3_mean_all,accuracy_mean_all = 0,0,0,0,0
    results = []

    # 用来累加单个学生的训练数据，最后求平均
    train_loss_mean_stu_all, v1_mean_stu_all, v2_mean_stu_all, v3_mean_stu_all, accuracy_mean_stu_all = 0, 0, 0, 0, 0
    rmse_all,auc_all = 0,0
    train_df = pd.DataFrame([],columns=['stu_id','epoch', 'loss', 'accuracy'])
    accuracy_val_all = 0
    rmse_val_all = 0
    auc_val_all = 0
    v2_val_all = 0
    v3_val_all = 0
    val_df = pd.DataFrame([],columns=[ 'epoch', 'accuracy_val', 'rmse_val', 'auc_val', 'v1_val', 'v2_val', 'v3_val'])
    # 修改一个人的数据多个epoch  dataloader与epoch同步
    while not train_data_loader.is_end():
      stu_id, kn_id, score_emb, user_k_kc, ex_id = train_data_loader.next_batch()
      # train_loss_mean_all, v1_mean_all, v2_mean_all, v3_mean_all, accuracy_mean_all = 0, 0, 0, 0, 0


      # 内存飙升
      model = KTModel(self.config.data_dim[self.config.DATASET], self.config)

      if self.train_conf.optimizer == 'SGD':
          optimizer = optim.SGD(
              model.parameters(),
              lr=self.train_conf.lr,
              momentum=self.train_conf.momentum,
              weight_decay=self.train_conf.wd)
      elif self.train_conf.optimizer == 'Adam':
          optimizer = optim.Adam(
              model.parameters(), lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
      else:
          raise ValueError("Non-supported optimizer!")

      # milestones为一个数组，如 [50,70]. gamma为倍数。如果learning rate开始为0.01 ，则当epoch为50时变为0.001，epoch 为70 时变为0.0001。
      # 当last_epoch=-1,设定为初始lr。  学习率策略
      lr_scheduler = optim.lr_scheduler.MultiStepLR(
          optimizer,
          milestones=self.train_conf.lr_decay_steps,
          gamma=self.train_conf.lr_decay)

      # reset gradient
      optimizer.zero_grad()
      accuracy_val_epoch = 0
      rmse_val_epoch = 0
      auc_val_epoch = 0
      v2_val_epoch = 0
      v3_val_epoch = 0

      train_loss_epoch, v1_epoch, v2_epoch, v3_epoch, accuracy_epoch = 0, 0, 0, 0, 0  # 用于计算数据集里的平均值

      for epoch in range(self.train_conf.max_epoch):

    # if not train_data_loader.is_end():
        iter_count = 0
        print('epoch begin:{}'.format(epoch + 1))
        print('第{}条数据'.format(train_data_loader.ptr ))

        model.train()

        '--------------------'
        # training
        # 通常用在epoch里面,但是不绝对，可以根据具体的需求来做。对lr进行调整。
        lr_scheduler.step()
        # 训练
        # for imgs, labels in train_loader:
        # train_data_loader.reset()
        # log =[]
        e = []  # 一个epoch 里每次训练的数  据
        train_loss_batch = 0
        iter_count += 1

        # print('while not train_data_loader.is_end():')

        # print('读取数据： debug ： stu_id:{} \r\nkn_id:{} \r\nscore_emb:{}\r\n user_k_kc:{}\r\n ex_id:{}\r\n'.format(stu_id, kn_id, score_emb, user_k_kc, ex_id))
        # print('读取数据： debug ：shape:   stu_id:{} \r\nkn_id:{} \r\nscore_emb:{}\r\n user_k_kc:{}\r\n ex_id:{}\r\n'.format(stu_id.shape, kn_id.shape, score_emb.shape, user_k_kc.shape, ex_id.shape))

        if self.use_gpu:
             stu_id, kn_id, score_emb, user_k_kc, ex_id = stu_id.cuda(), kn_id.cuda( ), score_emb.cuda( ), user_k_kc.cuda( ), ex_id.cuda( )

        # ??
        optimizer.zero_grad()

        # 初始化模型，并计算
        # train_loss, v1,v2,v3,state_last, diff_norm, grad , predict= model( stu_id, kn_id, score_emb, user_k_kc, ex_id,q_kn,d)
        state_last, state_2nd_last,predict,diff_norm= model( stu_id, kn_id, score_emb, user_k_kc, ex_id,q_kn,d)
        # grad计算

        train_loss,v1,v2,v3 = loss_func_temp(score_emb,predict,state_last,ex_id,q_kn)
        # train_loss_batch += train_loss
        # line10 的  l/u*
        grad_state_last = torch.autograd.grad(train_loss, state_last, retain_graph=True, create_graph=True)
        params = [pp for pp in model.parameters()]

        '''
        # 问题  公式1，3里的参数（）导数为0,最后一个梯度为None，params参数的顺序对应于模型中参数的定义位置（W，beiTa_2，beiTa_1）（另外一个点就是optimizer，应该没问题）
        # 根据论文，将参数传进RBP梯度计算算法中，即用Rbp替代了原本的loss.backwoard（）计算梯度的方式
        # 之后将计算出的梯度，自己放到对应的参数中
        '''
        grad = RBP(params,
                   #       求导时，有中间循环的过程吗
                   # 有：需要考虑中间的状态值，copy？怎么只是复制其值，导数会传 过来  ，grad的问题
                   # 没有：怎么将两个状态连起来      rbp里面时最后两个值的导数
                   [state_last],
                   [state_2nd_last],
                   grad_state_last,
                   # update_forward_diff=_update_forward_diff,
                   update_forward_diff=None,
                   # 论文中的 line 11， M
                   truncate_iter=self.config.model.truncate_iter,
                   rbp_method=self.config.model.grad_method)


        '''
        把梯度赋给对应的参数
        '''

        for pp, ww in zip(model.parameters(), grad):
          # print('gggg')
          # print(ww)
            pp.grad = ww
          # 最后一个为none
          # print(' pp.grad',pp.grad)
        pa = params[-2:]
        grad_pa = torch.autograd.grad(train_loss, pa, retain_graph=True, create_graph=True)
        for p, grad in zip(pa, grad_pa):
          p.grad = grad


        # print('参数调整之前')
        # t = 0
        # for p  in model.parameters():
        #   print(p)
        #   t+= 1

        # print('参数个数：',t)
        # 更新参数
        optimizer.step()
        # debug
        # print('参数调整之后')
        # for p in model.parameters():
        #   print(p)



        '''
        -----------------数据显示 --------------------------
        '''

        #
        v1_epoch += v1.item()
        v2_epoch += v2.item()
        v3_epoch += v3.item()
        train_loss_epoch += train_loss.item()
        # accuracy
        p = predict.clone().detach()
        p[predict.ge(0.5)] = 1
        p[predict.lt(0.5)] = 0
        # compute accuracy
        l = p - score_emb[ex_id]
        accuracy = torch.sum(l == 0) / l.numel()
        accuracy_epoch += accuracy.data.item()
        print('epoch epoch:{},acc:{},v2:{},v3:{}'.format(epoch+1,accuracy.data.item(),v2.item(),v3.item()))


        # train_loss = float(train_loss.data.cpu().numpy())

        # results['train_loss'] += [train_loss]
        # results['train_accuracy'] += [accuracy]
        # results['train_step'] += [iter_count]
        # print('epoch {}'.format(epoch))
        # display loss
        # display_accuracy += accuracy.data.item()

        accuracy_all += accuracy.data.item()
        # 显示一条数据的训练结果
        # if iter_count % self.train_conf.display_iter == 0:
        #   print("Loss @ epoch {:04d} iteration {:08d} = {}".format(
        #       # epoch + 1, iter_count  + 1, np.log10(train_loss)))
        #       epoch + 1, iter_count  ,train_loss))
        #   print("Acurracy @ epoch {:04d} iteration {:08d} = {}".format(
        #     epoch + 1, iter_count , accuracy))



          # tmp_key = 'diff_norm_{}'.format(iter_count + 1)
          # results[tmp_key] = diff_norm


    # w

      # 显示一次epoch的数据
        d_len = iter_count

        # tmp = {'train_loss': train_loss_epoch / d_len, 'v1': v1_epoch / d_len, 'v2': v2_epoch / d_len,
        #        'v3': v3_epoch / d_len,
        #        'train_accuracy': accuracy_epoch / d_len, 'iter_count': iter_count, 'diff_norm': str(diff_norm)}
        # e.append(tmp)
        #
        # train_loss_epoch_df = pd.DataFrame([[stu_id.item(),epoch + 1,train_loss_epoch / d_len,accuracy_epoch / d_len]], columns=['stu_id','epoch', 'loss','accuracy'])
        # train_df = train_df.append(train_loss_epoch_df,ignore_index=True)
        # log = {"epoch": epoch + 1, 'data': e}
        # print('训练结果：', json.dumps(log,sort_keys=True,indent=2))
        #
        # results.append(log)


      # 用来计算最后所有学生的总体的数据
      #   train_loss_mean_all += train_loss_epoch
      #   v1_mean_all += v1_epoch
      #   v2_mean_all +=  v2_epoch
      #   v3_mean_all += v3_epoch
      #   accuracy_mean_all += accuracy_epoch



        iter_count == 0   #epoch over
      # validation
      # if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:





        if (epoch + 1) % self.train_conf.valid_epoch == 0 :
            model.eval()
            # 验证
            # accuracy_val, rmse_val, auc_val, v1_val, v2_val, v3_val =validate(self,stu_id,val_data_loader,model,loss_func_temp,q_kn,d,epoch,val_df)
            val_df = validate(self,stu_id,val_data_loader,model,loss_func_temp,q_kn,d,epoch,val_df)
            # accuracy_val, rmse_val, auc_val, v1_val, v2_val, v3_val
            # accuracy_val_epoch += accuracy_val
            # rmse_val_epoch += rmse_all
            # auc_val_epoch += auc_val
            # v2_val_epoch +=  v2_val
            # v3_val_epoch += v3_val


      # snapshot model
        if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
            # logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
            print("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
            # xiugai baocun moxing
            # 设计一致性实验
            snapshot(model.module
                     if self.use_gpu else model, optimizer, self.config, epoch + 1 ,tag =  self.config.DATASET)

            # 一个epoch保存一次平均正确率
            with open('result/' + self.config.DATASET + '_train_accuracy.txt', 'a', encoding='utf8') as output_file:
                c = epoch+1
                a = accuracy_all/train_data_loader.get_len()
                output_file.write('epoch_'+ str(c) + '  Accuracy_mean: ' + str(a.tolist())   + '\n')
                accuracy_all = 0

    # d单stu删除model
      del model
      gc.collect()
      # val_tmp = self.train_conf.max_epoch//self.train_conf.valid_epoch
      # accuracy_val_all+= accuracy_val_epoch/val_tmp
      # rmse_val_all+= rmse_val_epoch/val_tmp
      # auc_val_all += auc_val_epoch/val_tmp
      # v2_val_all += v2_val_epoch/val_tmp
      # v3_val_all += v3_val_epoch/val_tmp

      rmse = torch.sqrt(torch.mean((score_emb[ex_id] - predict) ** 2))
      # compute AUC
      predict = predict.detach().numpy()
      score_emb = score_emb.detach().numpy()
      try:
          auc = roc_auc_score(score_emb[ex_id], predict)
      except ValueError:
          auc = 0  # 数据中有全对全错的问题
          pass
      # rmse =1
      # auc =1
      rmse_all += rmse
      auc_all += auc
      print('  epoch= %d, accuracy= %f, rmse= %f, auc= %f,v1,v2,v3 = %f,%f,%f' % (
          epoch + 1, accuracy, rmse, auc, v1, v2, v3))
    # 计算平均值
    # print('train_loss_mean:{},v1_mean:{},v2_mean:{},v3_mean{},accuracy:{}'.format((train_loss_mean_all/self.train_conf.max_epoch).tolist() ,(v1_mean_all/self.train_conf.max_epoch).tolist(),(v2_mean_all/self.train_conf.max_epoch).tolist(),(v3_mean_all/self.train_conf.max_epoch).tolist(),(accuracy_mean_all/self.train_conf.max_epoch).tolist()))
      print('train_loss_epoch_mean:{},v1_mean:{},v2_mean:{},v3_mean{},accuracy:{}'.format(train_loss_epoch/self.train_conf.max_epoch   , v1_epoch/self.train_conf.max_epoch , v2_epoch/self.train_conf.max_epoch ,v3_epoch/self.train_conf.max_epoch ,accuracy_epoch/self.train_conf.max_epoch ))
      # 累加单个学生的训练数据
      train_loss_mean_stu_all += train_loss_epoch/self.train_conf.max_epoch
      v1_mean_stu_all += v1_epoch/self.train_conf.max_epoch
      v2_mean_stu_all +=v2_epoch/self.train_conf.max_epoch
      v3_mean_stu_all += v3_epoch/self.train_conf.max_epoch
      accuracy_mean_stu_all += accuracy_epoch/self.train_conf.max_epoch

      # rmse = torch.sqrt(torch.mean((score_emb[ex_id] - predict) ** 2))
      # # compute AUC
      # predict = predict.detach().numpy()
      # score_emb = score_emb.detach().numpy()
      # try:
      #     auc = roc_auc_score(score_emb[ex_id], predict)
      # except ValueError:
      #     auc = 0  # 数据中有全对全错的问题
      #     pass
      # # rmse =1
      # # auc =1
      # print('  epoch= %d, accuracy= %f, rmse= %f, auc= %f,v1,v2,v3 = %f,%f,%f' % (
      #     epoch + 1, accuracy, rmse, auc, v1, v2, v3))

    # 保存所有的训练情况
    #   with open('result/' + self.config.DATASET + '_train_results.txt', 'w', encoding='utf8') as file:
    #    json.dump(results, file, indent=4, ensure_ascii=False)


    # l = train_data_loader.get_len()
    # accuracy_val_all /= l
    # rmse_val_all /= l
    # auc_val_all /= l
    # v2_val_all /= l
    # v3_val_all /= l
    # print('valid:mean:  accuracy_val_all:{}, rmse_val_all:{},auc_val_all:{},v2_val_all:{},v3_val_all:{}'.format(accuracy_val_all,rmse_val_all,auc_val_all,v2_val_all,v3_val_all))
    # pickle.dump(results,
    #             open(os.path.join(self.config.save_dir, self.config.DATASET + '_train_stats.p'), 'wb'))

  #  保存学生掌握状态
  #     with open('result/' + self.config.DATASET + '_' + str(stu_id)  +'_stu_stats.txt', 'w', encoding='utf8') as output_file:
  #      for item in model.U_embeding_origin:
  #         output_file.write(str(item.tolist()) + '\n')


    # 所有学生训练后，计算其平均值
    lamb = self.config.model.H
    epsilon = self.config.model.threshold
    train_df.to_csv('one_stu_result/' + self.config.DATASET +'/' + self.config.model.hp_type + '/hyper_lambda=' + str(lamb) + ',' + str(epsilon) +'.csv',index=False)
    # val_df.to_csv('one_stu_result/' + self.config.DATASET +'/' + self.config.model.hp_type + '/hyper_lambda=' + str(lamb) + ',' + str(epsilon) +'.csv',index=False)
    train_loss_mean_stu_all, v1_mean_stu_all, v2_mean_stu_all, v3_mean_stu_all, accuracy_mean_stu_all
    data_len = train_data_loader.get_len()
    print('所有学生训练后的平均值')
    print('train_loss_mean:{},v1_mean:{},v2_mean:{},v3_mean{},accuracy:{}'.format(
        train_loss_mean_stu_all/data_len,
        v1_mean_stu_all / data_len,
        v2_mean_stu_all / data_len,
        v3_mean_stu_all / data_len,
        accuracy_mean_stu_all / data_len))
    print('rmse_mean:{},auc_mean:{}'.format(rmse_all/data_len,auc_all/data_len))



def validate(self,stu_id,val_data_loader,model,loss_func_temp,q_kn, d,epoch,val_df):
        print('开始验证')

        # 选取stu——id的数据验证
        # val_data_loader = ValTestDataLoader(stu_id)

        # val_loss = []
        # val_counter = 0
        # '---------------------------'
        # # for imgs, labels in val_loader:
        # while not val_data_loader.is_end():
        if not val_data_loader.is_end():
            stu_id_val, kn_id_val, score_emb_val, user_k_kc_val, ex_id_val = val_data_loader.next_batch(stu_id_c = stu_id)

            # print('读取数据： debug ： stu_id:{} \r\nkn_id:{} \r\nscore_emb:{}\r\n user_k_kc:{}\r\n ex_id:{}\r\n'.format(stu_id, kn_id, score_emb, user_k_kc, ex_id))
            # print('读取数据： debug ：shape:   stu_id:{} \r\nkn_id:{} \r\nscore_emb:{}\r\n user_k_kc:{}\r\n ex_id:{}\r\n'.format(stu_id.shape, kn_id.shape, score_emb.shape, user_k_kc.shape, ex_id.shape))

            if self.use_gpu:
                stu_id_val, kn_id_val, score_emb_val, user_k_kc_val, ex_id_val = stu_id_val.cuda(), kn_id_val.cuda(), score_emb_val.cuda(), user_k_kc_val.cuda(), ex_id_val.cuda()

            state_last, state_2nd_last, predict_val, diff_norm = model(stu_id_val, kn_id_val, score_emb_val,
                                                                       user_k_kc_val, ex_id_val, q_kn, d)

            train_loss_val, v1_val, v2_val, v3_val = loss_func_temp(score_emb_val, predict_val, state_last, ex_id_val, q_kn)

            p = predict_val.clone().detach()
            p[predict_val.ge(0.5)] = 1
            p[predict_val.lt(0.5)] = 0
            # compute accuracy
            l = p - score_emb_val[ex_id_val]
            accuracy_val = torch.sum(l == 0) / l.numel()
            # compute RMSE

            rmse_val = torch.sqrt(torch.mean((score_emb_val[ex_id_val] - predict_val) ** 2))
            # compute AUC
            predict_val = predict_val.detach().numpy()
            score_emb_val = score_emb_val.detach().numpy()
            try:
                auc_val = roc_auc_score(score_emb_val[ex_id_val], predict_val)
            except ValueError:
                auc_val = 0  # 数据中有全对全错的问题
                pass
            # rmse =1
            # auc =1
            print('vaalid : epoch= %d, accuracy= %f, rmse= %f, auc= %f,v1,v2,v3 = %f,%f,%f' % (
            epoch + 1, accuracy_val, rmse_val, auc_val, v1_val, v2_val, v3_val))
            df = pd.DataFrame(
                [[epoch + 1, accuracy_val, rmse_val, auc_val, v1_val, v2_val, v3_val]],
                columns=[ 'epoch', 'accuracy_val', 'rmse_val', 'auc_val', 'v1_val', 'v2_val', 'v3_val'])
            val_df = val_df.append(df, ignore_index=True)
            return val_df
            # return  accuracy_val, rmse_val, auc_val, v1_val, v2_val, v3_val
            # with open('result/model_val.txt', 'a', encoding='utf8') as f:
            #   f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch + 1, accuracy, rmse, auc))
            #   方便后期使用pandas处理
            # with open('result/model_val_' + self.config.DATASET + '.txt', 'a', encoding='utf8') as f:
            #     # epoch= %d, accuracy= %f, rmse= %f, auc= %f
            #     f.write(' %d,  %f, %f,%f,%f,%f,%f,%f\n' % (epoch + 1, accuracy_val, rmse_val, auc_val, train_loss_val, v1_val, v2_val, v3_val))
            # re = {'epoch': epoch + 1, 'accuracy':accuracy.tolist(), 'rmse':rmse , 'auc':auc ,'train_loss':train_loss.tolist() , 'v1':v1.tolist(),'v2':v2.tolist(),'v3':v3.tolist()}
            # print(re)
            #
            #
            # with open('result/model_val_' +  self.config.DATASET + '.txt', 'a', encoding='utf8') as file:
            #   json.dump(re, file, indent=4, ensure_ascii=False)

