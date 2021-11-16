import numpy as np
import matplotlib
import  random as rand

matplotlib.use('Qt5Agg')
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator

'''
data_path:数据所在文件夹，是一个数组，  'hyper_result/math1/epsilon'
hp：变化的超参数，不同的超参走不同逻辑
'''
def impact_hyper(data_path,data_set,hp,values=[]):
    print()
    for d_name in data_set:
        for h in hp:
            data_path = 'hyper_result/'+ d_name +'/'+h
            if h == 'epsilon':
                data = pd.read_csv(data_path + '/hyper_lambda,epsilon=0.1,0.001.csv')
                # print(data)
                data.rename(columns={'loss': 'epsilon=0.001'}, inplace=True)

                data_epsilon_2 = pd.read_csv(data_path+ '/hyper_lambda,epsilon=0.1,0.01.csv')
                data_epsilon_2.rename(columns={'loss': 'epsilon=0.01'}, inplace=True)

                data_epsilon_3 = pd.read_csv(data_path+'/hyper_lambda,epsilon=0.1,0.1.csv')
                data_epsilon_3.rename(columns={'loss': 'epsilon=0.1'}, inplace=True)

                data_epsilon_4 = pd.read_csv(data_path+'/hyper_lambda,epsilon=0.1,0.5.csv')
                data_epsilon_4.rename(columns={'loss': 'epsilon=0.5'}, inplace=True)

                data = pd.merge(data, data_epsilon_2, on='epoch')
                data = pd.merge(data, data_epsilon_3, on='epoch')
                data = pd.merge(data, data_epsilon_4, on='epoch')
                # style = ['-', '--', '-.', ':']
                # print(data)
                ax = data.plot(x='epoch', y=['epsilon=0.001', 'epsilon=0.01', 'epsilon=0.1', 'epsilon=0.5'], grid=True ,style =['-', '--', '-.', ':'],legend =['1','2','3','4'])

                ax.grid(linestyle="--", alpha=0.3)
                ax.legend(labels = [r'$\epsilon$' + '=0.001',r'$\epsilon$' + '=0.01',r'$\epsilon$' + '=0.1',r'$\epsilon$' + '=0.5'])

                # ax.set_title("the convergence ratio of epsilon  on   "   + d_name)
                ax.set_ylabel('loss')


                # 设置间隔为5
                ax.xaxis.set_major_locator(MultipleLocator(5))
                # plt.show()
                # l = ['epsilon=0.001', 'epsilon=0.01', 'epsilon=0.1', 'epsilon=0.5']
               # ax_acc.xaxis.set_minor_locator(MultipleLocator(0.1))
               #  plt.show()
                print()
                #
                fig = ax.get_figure()
                # fig_2 = ax_acc.get_figure()
                fig.savefig('hyper_result/loss_'+h +'_' + d_name + '.png')
                # fig_2.savefig('hyper_result/accuracy_'+h +'_' + d_name + '.png')
                # plt.show()

            #     lambda的计算
            else:
                # print()
                data = pd.read_csv(data_path + '/hyper_lambda,epsilon=0.001,0.001.csv')
                data.rename(columns={'loss': 'lambda=0.001'}, inplace=True)

                data_epsilon_2 = pd.read_csv(data_path + '/hyper_lambda,epsilon=0.1,0.001.csv')
                data_epsilon_2.rename(columns={'loss': 'lambda=0.1'}, inplace=True)

                data_epsilon_3 = pd.read_csv(data_path + '/hyper_lambda,epsilon=0.5,0.001.csv')

                data_epsilon_3.rename(columns={'loss': 'lambda=0.5'}, inplace=True)

                data_epsilon_4 = pd.read_csv(data_path + '/hyper_lambda,epsilon=1,0.001.csv')

                data_epsilon_4.rename(columns={'loss': 'lambda=1'}, inplace=True)

                data_epsilon_5 = pd.read_csv(data_path + '/hyper_lambda,epsilon=5,0.001.csv')
                data_epsilon_5.rename(columns={'loss': 'lambda=5'}, inplace=True)

                data_epsilon_6 = pd.read_csv(data_path + '/hyper_lambda,epsilon=10,0.001.csv')
                data_epsilon_6.rename(columns={'loss': 'lambda=10'}, inplace=True)

                data_epsilon_7 = pd.read_csv(data_path + '/hyper_lambda,epsilon=15,0.001.csv')
                data_epsilon_7.rename(columns={'loss': 'lambda=15'}, inplace=True)

                data_epsilon_8 = pd.read_csv(data_path + '/hyper_lambda,epsilon=20,0.001.csv')
                data_epsilon_8.rename(columns={'loss': 'lambda=20'}, inplace=True)

                data = pd.merge(data, data_epsilon_2, on='epoch')
                data = pd.merge(data, data_epsilon_3, on='epoch')
                data = pd.merge(data, data_epsilon_4, on='epoch')
                data = pd.merge(data, data_epsilon_5, on='epoch')
                data = pd.merge(data, data_epsilon_6, on='epoch')
                data = pd.merge(data, data_epsilon_7, on='epoch')
                data = pd.merge(data, data_epsilon_8, on='epoch')
                # print(data)
                # ax = data.plot(x='epoch', y=['lambda=0.001', 'lambda=0.1', 'lambda=0.5', 'lambda=1', 'lambda=5', 'lambda=10', 'lambda=15', 'lambda=20'], grid=True)
                linestyle_str = [
                    ('solid', 'solid'),  # Same as (0, ()) or '-'
                    ('dotted', 'dotted'),  # Same as (0, (1, 1)) or '.'
                    ('dashed', 'dashed'),  # Same as '--'
                    ('dashdot', 'dashdot')]  # Same as '-.'

                linestyle_tuple = [
                    ('loosely dotted', (0, (1, 10))),
                    ('dotted', (0, (1, 1))),
                    ('densely dotted', (0, (1, 1))),

                    ('loosely dashed', (0, (5, 10))),
                    ('dashed', (0, (5, 5))),
                    ('densely dashed', (0, (5, 1))),

                    ('loosely dashdotted', (0, (3, 10, 1, 10))),
                    ('dashdotted', (0, (3, 5, 1, 5))),
                    ('densely dashdotted', (0, (3, 1, 1, 1))),

                    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
                    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

                # style =[ (0, (3, 10, 1, 10, 1, 10)),(0, (3, 10, 1, 10)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 10)),'-', '--', '-.', ':']
                style =[ (0, (3, 5, 1, 5, 1, 5)),(0, (5, 5)),(0, (3, 1, 1, 1, 1, 1)),(0, (1, 10)),'-', '--', '-.', ':']
                column = ['lambda=0.001', 'lambda=0.1', 'lambda=0.5', 'lambda=1', 'lambda=5', 'lambda=10', 'lambda=15', 'lambda=20']

                    # ax.annotate(repr(i))

                for s, c in zip(style, column):
                    # ax_acc = data.plot(x='epsilon', y=d, grid=True, marker=m)
                    print(s)
                    plt.plot(data['epoch'], data[c], linestyle=s)

                plt.legend(loc="upper right", borderpad = 30)
                # plt.grid(linestyle="--", alpha=0.3)
                # print()
                ax = plt.gca()



                ax.grid(linestyle="--", alpha=0.3)
                print()
                print()
                # ax.set_title("the convergence ratio of lambda  on   " + d_name)
                ax.set_ylabel('loss')
                # 设置legend
                lab = []
                d = [0.001,0.1,0.5,1,5,10,15,20]
                for i in d:
                  # ax.legend(labels = [r'$\epsilon$' + '=0.001',r'$\epsilon$' + '=0.01',r'$\epsilon$' + '=0.1',r'$\epsilon$' + '=0.5'])
                  lab.append(r'$\lambda$=' + str(i))
                ax.legend(labels = lab)
                # plt.show()
                # print()
                # 设置间隔为5
                ax.xaxis.set_major_locator(MultipleLocator(5))
                # plt.show()

                fig = ax.get_figure()
                # fig_2 = ax_acc.get_figure()
                fig.savefig('hyper_result/loss_' + h + '_' + d_name + '.png')
                # fig_2.savefig('hyper_result/accuracy_' + h + '_' + d_name + '.png')
                # plt.sho、w()


def plot_hyper_inconsienty():
    lambda_data = {'lambda':[0.01,0.1,0.5,1,5],
                   'Assist2009':[0.169,0.179,0.190,0.210,0.215],
                   'Assist2015':[0.054,0.052,0.053,0.055,0.053],
                   'Math1':[0.055,0.048,0.045,0.061,0.075],
                   'Math2':[0.053,0.037,0.035,0.048,0.051]}
    lambda_df = pd.DataFrame(lambda_data)
    ax_lambda = lambda_df.plot(x='lambda',
                   y=['Assist2009', 'Assist2015', 'Math1', 'Math2'], style=['-', '--', '-.', ':'],grid=True)

    ax_lambda.grid(linestyle="--", alpha=0.3)
    # ax_lambda.set_title("Inconsistency Sensitivity of lambda"  )
    ax_lambda.set_ylabel('Inconsistency Sensitivity')
    ax_lambda.set_xlabel(r'$\lambda$')
    # 并无太大区别？？
    epsilon_data = {'epsilon': [0.001, 0.01, 0.1,0.5],
                   'Assist2009': [0.169, 0.171, 0.173, 0.186],
                   'Assist2015': [0.052, 0.054, 0.053,0.053],
                   'Math1': [0.045, 0.053, 0.063, 0.086],
                   'Math2': [0.037, 0.041, 0.051, 0.064]}

    epsiolon_df = pd.DataFrame(epsilon_data)
    ax_epsilon = epsiolon_df.plot(x='epsilon',
                        y=['Assist2009', 'Assist2015', 'Math1', 'Math2'], style=['-', '--', '-.', ':'],grid=True)

    ax_epsilon.grid(linestyle="--", alpha=0.3)
    # ax_epsilon.set_title("Inconsistency Sensitivity of Epsilon"  )
    ax_epsilon.set_ylabel('Inconsistency Sensitivity')
    ax_epsilon.set_xlabel(r'$\epsilon$')

    plt.show()
    fig = ax_lambda.get_figure()
    fig_2 = ax_epsilon.get_figure()
    fig.savefig('hyper_result/Inconsistency Sensitivity of Lambda.png')
    fig_2.savefig('hyper_result/Inconsistency Sensitivity of Epsilon.png')

    # print(lambda_df)
def plot_accuracy():
    dataset = ['math1', 'math2', 'assist2015', 'assist2009']
    type = ['epsilon', 'lambda']
    for h in type:
        data = pd.DataFrame([],columns=[h,'accuracy'])
        for d_name in dataset:
            if len(data.index) == 0:
                data = pd.read_csv('hyper_result/accuracy_' + h + '_' + d_name + '.csv')
                data.rename(columns={'accuracy':d_name},inplace = True)
                continue
            acc_data = pd.read_csv('hyper_result/accuracy_' + h + '_' + d_name + '.csv')
            acc_data.rename(columns={'accuracy': d_name }, inplace=True)
            # print(acc_data)
            data =  pd.merge(acc_data, data, on='epsilon')

        print(data)
        print()
        # 折点样式
        markers = ['o','v','D','s']
        # markers = ['o','v','2','v']
        # markers = ['1','2','3','4']
        d_set =['assist2009','assist2015','math1','math2']
        for m,d in zip(markers,d_set):

            # ax_acc = data.plot(x='epsilon', y=d, grid=True, marker=m)
            plt.plot(data['epsilon'], data[d] , marker=m,label=d)

        # ax_acc = data.plot(x='epsilon', y=['assist2009','assist2015','math1','math2'], grid=True, marker=['o','v','v','v'])

        plt.legend(  loc="upper right")
        plt.grid(linestyle="--", alpha=0.3)
        # print()
        ax_acc = plt.gca()
        # ax_acc.set_title("the ACCURACY of lambda  on   " + d_name)
        ax_acc.set_ylabel('accuracy')
        if h == 'epsilon':
            # plt.xlabel(r'$\epsilon$')
            ax_acc.set_xlabel(r'$\epsilon$')
        else:
            # plt.xlabel(r'$\lambda$')
            ax_acc.set_xlabel(r'$\lambda$')
        plt.show()
        # print()
        fig = ax_acc.get_figure()
        fig.savefig('hyper_result/accuracy_' + h +  '.png')


# plot_accuracy()
# plot_hyper_inconsienty()

# plot  loss
# dataset = ['math1', 'math2', 'assist2015','assist2009']
# type = ['epsilon', 'lambda']
# impact_hyper(data_path='',data_set = dataset,hp= type,values=[])
# # pd.read_csv


