# coding=utf-8
#Author: chengsong
#Time: 2019-01-15

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import time
plt.rc('font', family='SimHei', size=13)
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from math import log
# 多线程
from multiprocessing.pool import ThreadPool
from multiprocessing.dummy import Pool
import multiprocessing

# CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET
# logging.debug('This is debug message')
# logging.info('This is info message')
# logging.warning('This is warning message')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S')

# Y变量统一默认为"y"，设置为其他会报错
class logistic_score_card(object):

    def __init__(self,
                 max_deeps=2,
                 min_info_gain_epos=0.001,
                 multiprocessing=0,
                 base_score=600,
                 increase_score = 50,
                 min_iv = 0.02,
                 max_corr = 0.8,
                 C = 0.01,
                 penalty = 'l1'
                 ):  # 一些最终的函数结果值

        self.min_iv = min_iv #选取IV值的标准（选取IV值大于0.1的值）
        self.max_corr = max_corr #选取相关性的标准（选取相关性小于0.8的值）
        self.C = C #筛选变量时的逻辑回归选取的参数
        self.penalty = penalty #筛选变量时的逻辑回归选取的参数

        self.data = []

        self.max_deeps = max_deeps  # 决策树最大深度
        self.min_info_gain_epos = min_info_gain_epos  # 决策树最小信息熵差
        self.multiprocessing=multiprocessing #默认为0：不开启多进程；1为开启多进程

        self.col_type = []  # 不计算IV值，仅为区分变量为离散还是连续（1：连续；2：离散）
        self.col_type_iv = []  # 计算IV值（不离散化）（连续变量离散化后IV值基本不大，为节省时间可参考不离散化的IV值）
        self.col_type_cut_iv = []  # 计算连续变量离散化后的IV值

        self.cut_points_list = []  # 连续变量的切分点，按小于等于，大于切分。（如有空值返回为'null'）
        self.cut_points_result = []  # 连续变量的切分点，按小于等于，大于切分。（如有空值返回为'null'）例如:[col,['null',3,7]]

        self.col_result = [] #最终评分卡选择的变量
        self.logistic_reuslt = {}  # 逻辑回归模型训练结果
        self.base_score = base_score #基础分
        self.increase_score = increase_score #增加分（概率增加一份所增加的分数）
        self.score_card_parameter={} #评分卡参数
        self.score_card = [] #评分卡结果
        self.logistic_auc_ks = {} #模型AUC KS值

    def check_data_y(self,data):
        '''
        检查数据结构，数据预测变量为0,1，并以“y”命名

        :param data:
        :return:
        '''

        if 'y' not in data.columns:
            logging.ERROR('未检测到"y"变量，请将预测变量命名改为"y"')

    # 读取原始数据
    # 划分训练集和测试集
    def get_data_train_test(self,data,test_size=0.25,random_state=1234):
        self.check_data_y(data)
        x_train, x_test, y_train, y_test = train_test_split(data[[col for col in data.columns if col !='y']], data['y'], test_size=test_size,
                                                            random_state=random_state)  # 随机选择25%作为测试集，剩余作为训练集
        data_train = x_train.reset_index()
        del data_train['index']
        data_train['y'] = y_train.reset_index()['y']
        data_test = x_test.reset_index()
        del data_test['index']
        data_test['y'] = y_test.reset_index()['y']
        return data_train, data_test

    def save_data(self, data):
        self.data = data

    def del_data(self):
        self.data = ''

    # 划分离散和连续变量（连续变量：int64,float64；离散变量：其他）
    def get_col_discrete_continue(self, data):
        logging.info('连续和离散变量划分中。。。')
        col_all = data.columns
        col_all_type = data.dtypes

        for i in range(len(col_all)):
            if col_all[i] != 'y':
                if str(col_all_type[i]) in ('int64', 'float64'):
                    self.col_type.append([col_all[i], 1])
                else:
                    self.col_type.append([col_all[i], 0])
        logging.info('连续和离散变量划分完成！')

    # ----------------------------------------最优分组（决策树ID3算法（信息熵））--------------------------------------------------------------------

    # 计算信息熵
    def get_info_entropy(self, data):
        p1 = len(data[data['y'] == 1]) / len(data)
        if p1 == 1 or p1 == 0:
            info_entropy = 0
        else:
            info_entropy = -p1 * log(p1) - (1 - p1) * log(1 - p1)
        return info_entropy

    def get_cut_not_null(self, data_not_null, col):
        cut_point = '无最优切分点'
        max_info_gain = self.min_info_gain_epos
        # for i in tqdm(np.sort(list(data_not_null[col].unique()), axis=0)[:-1]):  # 最后一位不切分（带进度条）
        for i in np.sort(list(data_not_null[col].unique()), axis=0)[:-1]:  # 最后一位不切分
            data_1 = data_not_null[data_not_null[col] <= i]  # 切分点为小于等于
            data_2 = data_not_null[data_not_null[col] > i]
            data_1_p = len(data_1) / len(data_not_null)
            data_2_p = len(data_2) / len(data_not_null)
            info_gain = self.get_info_entropy(data_not_null) - data_1_p * self.get_info_entropy(
                data_1) - data_2_p * self.get_info_entropy(data_2)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                cut_point = i
        return cut_point, max_info_gain

    # 最优分组（决策树ID3算法（信息熵））
    def get_cut_all_not_null(self, data_not_null, col, deeps=2):
        if deeps > 0:
            cut_point, max_info_gain = self.get_cut_not_null(data_not_null, col)
            if cut_point != '无最优切分点':
                self.cut_points_list.append([col, cut_point])
                data_1 = data_not_null[data_not_null[col] <= cut_point]
                data_2 = data_not_null[data_not_null[col] > cut_point]
                if len(data_1[col].unique()) > 1:  # 数据最少2行
                    self.get_cut_all_not_null(data_1, col, deeps - 1)
                if len(data_2[col].unique()) > 1:  # 数据最少2行
                    self.get_cut_all_not_null(data_2, col, deeps - 1)


    # 排序切分点以及展示样式
    def transform_cut_points_list(self):
        # 排序切分点
        self.cut_points_list = [[i, j] for i, j in
                                pd.DataFrame(self.cut_points_list, columns=['col', 'col_cut']).sort_values(
                                    by=['col', 'col_cut']).values]
        for i in pd.DataFrame(self.cut_points_list, columns=['col', 'col_cut'])['col'].unique():
            col_cuts = []
            for col, col_cut in self.cut_points_list:
                if col == i:
                    col_cuts.append(col_cut)
            self.cut_points_result.append([i, col_cuts])

    ##切分连续变量（决策树ID3算法（信息熵））
    def get_continue_cut(self, data):
        logging.info('连续变量最优分组进行中。。。')
        for col, col_type in tqdm(self.col_type):
            if col_type == 1:
                data_col=data[[col,'y']].copy()
                data_not_null = data_col.dropna(subset=[col])
                if len(data_not_null) != len(data_col):
                    self.cut_points_list.append([col, 'null'])
                    self.get_cut_all_not_null(data_not_null, col,self.max_deeps)
                else:
                    self.get_cut_all_not_null(data_not_null, col,self.max_deeps)
        self.transform_cut_points_list()
        logging.info('连续变量最优分组完成！')

    # ------------------------------------------------------------------------------------------------------------
    def get_cut_all_not_null_multiprocessing(self, col, deeps=2):
        data_not_null = self.data[[col,'y']].dropna(subset=[col])
        if deeps > 0:
            cut_point, max_info_gain = self.get_cut_not_null(data_not_null, col)
            if cut_point != '无最优切分点':
                self.cut_points_list.append([col, cut_point])
                data_1 = data_not_null[data_not_null[col] <= cut_point]
                data_2 = data_not_null[data_not_null[col] > cut_point]
                if len(data_1[col].unique()) > 1:  # 数据最少2行
                    self.get_cut_all_not_null(data_1, col, deeps - 1)
                if len(data_2[col].unique()) > 1:  # 数据最少2行
                    self.get_cut_all_not_null(data_2, col, deeps - 1)

    def get_continue_cut_multiprocessing(self, data, multiprocessing_type=1): #默认1为多进程，其他为多线程
        logging.info('多进程版-连续变量最优分组进行中。。。')
        self.save_data(data)
        if multiprocessing_type == 1:
            logging.info('已启用多进程，最优分箱进行中。。。')
            pool = Pool(multiprocessing.cpu_count())  # 设置进程数一般为cpu数量
        else:
            logging.info('已启用多线程，最优分箱进行中。。。')
            pool = ThreadPool(multiprocessing.cpu_count() * 2)  # 设置线程数一般为cpu的2倍
        cols = [col for col, col_type in self.col_type if col_type == 1]
        # pool.imap_unordered(self.get_cut_all_not_null_multiprocessing, cols)
        for i in tqdm(pool.imap_unordered(self.get_cut_all_not_null_multiprocessing, cols), total=len(cols),
                      leave=False):
            pass
        pool.close()
        pool.join()
        self.transform_cut_points_list()

        self.del_data()
        logging.info('多进程版-连续变量最优分组完成！')

    # ------------------------------------------------------------------------------------------------------------

    # 计算IV值（不离散化）（连续变量离散化后IV值有明显差别，为节省时间仅可参考不离散化的IV值）
    def get_col_type_iv(self, data):
        data = data.fillna('空值')
        for col, type in tqdm(self.col_type):
            iv = 0
            for i in data[col].unique():
                p1 = (np.sum((data[col] == i) & (data['y'] == 1))) / np.sum(data['y'] == 1)
                p0 = (np.sum((data[col] == i) & (data['y'] == 0))) / np.sum(data['y'] == 0)
                if p1 == 0 or p0 == 0:
                    iv += 0
                else:
                    iv += (p1 - p0) * np.log(p1 / p0)
            self.col_type_iv.append([col, type, iv])  # 返回变量名和IV值

    # 根据cut离散化连续变量(按1增加，空值不划分默认为空)
    def get_cut_result(self, data,cut_points_result):
        logging.info('根据cut离散化连续变量进行中。。。')
        data_cut_result = data.copy()
        for col, cut_points in tqdm(cut_points_result):
            data_col=data[[col]]
            cluster = 1
            data_cut_result.loc[data_col[col].notnull(),col] = str(cluster)
            for cut_point in cut_points:
                if cut_point != 'null':
                    cluster += 1
                    data_cut_result.loc[data_col[col] > cut_point, col] = str(cluster)
        logging.info('根据cut离散化连续变量完成！')
        return data_cut_result

    # 计算IV值（计算离散化后的IV值）
    def get_col_type_cut_iv(self, data):
        logging.info('计算所有变量IV值进行中。。。')
        data = data.fillna('空值')
        for col, type in tqdm(self.col_type):
            iv = 0
            for i in data[col].unique():
                p1 = (np.sum((data[col] == i) & (data['y'] == 1))) / np.sum(data['y'] == 1)
                p0 = (np.sum((data[col] == i) & (data['y'] == 0))) / np.sum(data['y'] == 0)
                if p1 == 0 or p0 == 0:
                    iv += 0
                else:
                    iv += (p1 - p0) * np.log(p1 / p0)
            self.col_type_cut_iv.append([col, type, iv])  # 返回变量名和IV值
        logging.info('计算所有变量IV值完成！')

    # 离散变量转化为WOE值
    def get_transform_woe(self, data):
        logging.info('数据WOE转化进行中。。。')
        data = data.fillna('空值')
        data_woe = data.copy()
        for col in tqdm([col for col in data.columns if col!='y']):
            data_col=data[[col,'y']]
            for cluster in data_col[col].unique():
                total_1 = len(data_col[data_col['y'] == 1])
                total_0 = len(data_col[data_col['y'] == 0])
                cluster_1 = len(data_col[(data_col[col] == cluster) & (data_col['y'] == 1)])
                cluster_0 = len(data_col[(data_col[col] == cluster) & (data_col['y'] == 0)])
                if cluster_1 > 0 and cluster_0 > 0:
                    woe = log(((cluster_0) / total_0) / ((cluster_1) / total_1))
                else:
                    woe = 0
                data_woe.loc[data_col[col] == cluster, col] = woe
        data_woe = data_woe.astype(float)  # 转化为数值型
        logging.info('数据WOE转化完成')
        return data_woe

    # 通过IV值和相关性选择变量（默认参数提出IV值<0.1，相关系数>0.8的变量）
    def get_iv_corr_col(self, data_woe, min_iv=0.1, max_corr=0.8):
        logging.info('根据IV值大于 %s 且 相关性小于 %s 选取变量中。。。' % (min_iv, max_corr))
        col_type_cut_iv = pd.DataFrame(self.col_type_cut_iv, columns=['col', 'col_type', 'col_iv'])
        col_result = col_type_cut_iv['col'][col_type_cut_iv['col_iv'] >= min_iv]
        data_woe_corr = data_woe[col_result].corr()
        col_delete = []
        for col_1 in tqdm(data_woe_corr.columns):
            for col_2 in data_woe_corr[data_woe_corr[col_1] >= max_corr].index:
                if col_1 != col_2:
                    col_1_iv = list(col_type_cut_iv['col_iv'][col_type_cut_iv['col'] == col_1])[0]
                    col_2_iv = list(col_type_cut_iv['col_iv'][col_type_cut_iv['col'] == col_2])[0]
                    if col_1_iv <= col_2_iv:
                        col_delete.append(col_1)
        col_result = [col for col in col_result if col not in list(np.unique(col_delete))]
        logging.info('变量选取完成，总共 %s 个变量，最终筛选出 %s 个变量' % (len(data_woe.columns)-1,len(col_result)))
        return col_result

    # 通过IV值和相关性以及逻辑回归L1选择变量（默认参数提出IV值>=0.1，相关系数>0.8,l1正则筛选的变量）
    def get_iv_corr_logistic_l1_col(self, data_woe, min_iv=0.1, max_corr=0.8,C=0.01,penalty='l1'):
        logging.info('根据IV值大于 %s 且 相关性小于 %s ，以及l1正则选取变量进行中。。。' % (min_iv, max_corr))
        col_type_cut_iv = pd.DataFrame(self.col_type_cut_iv, columns=['col', 'col_type', 'col_iv'])
        col_result = col_type_cut_iv['col'][col_type_cut_iv['col_iv'] >= min_iv]
        data_woe_corr = data_woe[col_result].corr()
        col_delete = []
        for col_1 in tqdm(data_woe_corr.columns):
            for col_2 in data_woe_corr[data_woe_corr[col_1] >= max_corr].index:
                if col_1 != col_2:
                    col_1_iv = list(col_type_cut_iv['col_iv'][col_type_cut_iv['col'] == col_1])[0]
                    col_2_iv = list(col_type_cut_iv['col_iv'][col_type_cut_iv['col'] == col_2])[0]
                    if col_1_iv <= col_2_iv:
                        col_delete.append(col_1)
        col_result = [col for col in col_result if col not in list(np.unique(col_delete))]

        lr = linear_model.LogisticRegression(C=C, penalty=penalty).fit(data_woe[col_result], data_woe['y'])
        col_result=[col_result[i] for i in range(len(col_result)) if lr.coef_[0][i] != 0]

        logging.info('变量选取完成，总共 %s 个变量，最终筛选出 %s 个变量' % (len(data_woe.columns)-1,len(col_result)))
        return col_result

    # -------------------------------------------logistic模型训练-----------------------------------------------------------------

    def get_auc_ks(self, data):
        data_x = data[[i for i in data.columns if i != 'y']]
        data_y = data['y']
        predictions = self.logistic_reuslt.predict_proba(data_x)  # 每一类的概率
        false_positive_rate, recall, thresholds = roc_curve(data_y, predictions[:, 1])
        roc_auc = auc(false_positive_rate, recall)
        ks = max(recall - false_positive_rate)
        return roc_auc, ks

    def get_cut_str(self,col_cuts_test):
        col_result = []
        for col, cut in col_cuts_test:
            num = 1
            if 'null' in cut:
                col_result.append([col, 'null', 'null'])
            cut = [i for i in cut if i != 'null']
            col_result.append([col, str(num), '(-,' + str(cut[0]) + ']'])
            for i in range(len(cut) - 1):
                num += 1
                if cut[i] != 'null':
                    cut_detail = '(' + str(cut[i]) + ',' + str(cut[i + 1]) + ']'
                    col_result.append([col, str(num), cut_detail])
            col_result.append([col, str(len(cut) + 1), '(' + str(cut[-1]) + ',+)'])
        result = pd.DataFrame(col_result)
        result.columns = ['col', 'cut_point', 'cut_str']
        return result


    def get_logistic_socre_card(self,data,col_result,cut_points_result,increase_score=50, base_score=600):
        logging.info('制作评分卡进行中。。。')

        cut_points_result=[i for i in cut_points_result if i[0] in col_result]
        data_result = data[col_result].copy()
        data_result['y'] = data['y']
        get_cut_result = self.get_cut_result(data_result,cut_points_result)
        data_woe = self.get_transform_woe(get_cut_result)  # 根据切分点将数据woe化

        lr=linear_model.LogisticRegression(C=1, penalty='l2')
        lr.fit(data_woe[col_result], data_woe['y'])

        b = -increase_score / np.log(2)
        a = base_score - lr.intercept_[0] * b

        # 评分表
        result = pd.DataFrame()
        for col in [i for i in get_cut_result.columns if i != 'y']:
            data_col = pd.DataFrame(data=[get_cut_result[col], data_woe[col]]).T
            data_col['col'] = col
            data_col.columns = ['cut_point', 'woe', 'col']
            result = pd.concat([result, data_col])
        result = result.drop_duplicates()

        score_card = []
        for cut_point, woe, col in result.values:
            woe_num_rate = len(data_woe[data_woe[col] == woe]) / len(data_woe)
            woe_y_rate = len(data_woe[(data_woe[col] == woe) & (data_woe['y'] == 1)]) / len(data_woe[data_woe[col] == woe])
            score_card.append([col, cut_point, woe, woe_num_rate, woe_y_rate])
        score_card_result = pd.DataFrame(score_card,
                                         columns=['col', 'cut_point', 'woe', 'woe_num_rate', 'woe_y_rate']).sort_values(
            by=['col', 'cut_point'])
        col_coef = pd.DataFrame(col_result, lr.coef_[0]).reset_index()
        col_coef.columns = ['col_coef', 'col']
        score_card_result_coef = pd.merge(score_card_result, col_coef, on=['col'], how='left')
        score_card_result_coef['score'] = score_card_result_coef['woe'] * score_card_result_coef['col_coef'] * b

        col_cuts_str = self.get_cut_str(cut_points_result)
        score_card_result_coef_not_null = score_card_result_coef.fillna('null')
        score_card_result_coef_cut = pd.merge(score_card_result_coef_not_null, col_cuts_str, on=['col', 'cut_point'],
                                              how='left')
        score_card_result_coef_cut.loc[score_card_result_coef_cut['cut_str'].isnull(), 'cut_str'] = \
            score_card_result_coef_cut['cut_point'][score_card_result_coef_cut['cut_str'].isnull()]

        score_card_result_coef_cut_iv = pd.merge(score_card_result_coef_cut,
                                                 pd.DataFrame(self.col_type_cut_iv, columns=['col', 'col_type', 'iv']),
                                                 on=['col'], how='left')

        score_card = score_card_result_coef_cut_iv[
            ['col', 'col_type', 'iv', 'cut_point', 'cut_str', 'woe', 'col_coef', 'score', 'woe_num_rate', 'woe_y_rate']]
        score_card['lr_intercept']=lr.intercept_[0]
        score_card=score_card[['col', 'col_type', 'iv', 'cut_point', 'cut_str', 'woe', 'col_coef','lr_intercept','score', 'woe_num_rate', 'woe_y_rate']]
        # self.score_card = score_card_result_coef_cut_iv[['变量名','变量类型','iv值','切分点','切分区间','woe值','logistic模型coef参数','logistic模型intercept参数','得分','用户占比','逾期率']]

        logging.info('制作评分卡完成')
        return score_card

    # ------------------------------------------------------------------------------------------------------------

    # 运行会得到训练数据
    def fit(self, data):
        logging.info('任务开始。。。')
        self.check_data_y(data)

        self.get_col_discrete_continue(data)  # 划分离散和连续变量
        if self.multiprocessing==0:
            self.get_continue_cut(data)  # 连续变量最优分组
        else:
            self.get_continue_cut_multiprocessing(data)  # 连续变量最优分组（多进程版）

        get_cut_result = self.get_cut_result(data,self.cut_points_result) #按切分点划分数据，得到全部的离散数据

        self.get_col_type_cut_iv(get_cut_result)  # 按切分点切分数据，求所有变量IV值
        data_woe = self.get_transform_woe(get_cut_result)  # 根据切分点讲数据woe化

        self.col_result=self.get_iv_corr_logistic_l1_col(data_woe, min_iv=self.min_iv, max_corr=self.max_corr, C=self.C, penalty=self.penalty)

        self.logistic_reuslt = linear_model.LogisticRegression(C=1,penalty='l2')
        self.logistic_reuslt.fit(data_woe[self.col_result], data_woe['y'])

        data_woe_result = data_woe[self.col_result].copy()
        data_woe_result['y'] = data_woe['y'].copy()

        auc, ks = self.get_auc_ks(data_woe_result)

        self.logistic_auc_ks['train'] = {}
        self.logistic_auc_ks['train']['auc'] = auc
        self.logistic_auc_ks['train']['ks'] = ks

        cut_points_result=[i for i in self.cut_points_result if i[0] in self.col_result]
        self.score_card=self.get_logistic_socre_card(data,self.col_result,cut_points_result)
        logging.info('任务结束')

    def predict_score_proba(self, data,cut_points_result,score_card,increase_score=50, base_score=600):
        logging.info('预测用户分数中。。。')
        b = -increase_score / np.log(2)
        a = base_score - score_card['lr_intercept'][0] * b
        col_result = score_card['col'].unique().tolist()
        data=data[col_result].copy()
        cut_points_result=[i for i in cut_points_result if i[0] in col_result]
        data_cut_points = self.get_cut_result(data, cut_points_result).fillna('null')
        data_score_proba = data_cut_points.copy()
        for col in data_cut_points.columns:
            data_cut_points_col=data_cut_points[[col]]
            for cut_point in data_cut_points_col[col].unique():
                cut_point_score=list(score_card['score'][(score_card['col'] == col) & (score_card['cut_point'] == cut_point)])
                if len(cut_point_score)>0:
                    data_score_proba.loc[data_cut_points_col[col]==cut_point, col] = cut_point_score[0]
                else:
                    data_score_proba.loc[data_cut_points_col[col] == cut_point, col] = 0
        data_score_proba['total_score'] = data_score_proba.apply(lambda x: x.sum()+600, axis=1)
        data_score_proba['proba']=1-1/(1+np.e**((data_score_proba['total_score']-a)/b))
        logging.info('预测用户分数完成')
        return data_score_proba

    def score(self,data,cut_points_result,score_card):
        data_score_proba=self.predict_score_proba(data,cut_points_result,score_card)
        false_positive_rate, recall, thresholds = roc_curve(data['y'], data_score_proba['proba'])
        roc_auc = auc(false_positive_rate, recall)
        ks = max(recall - false_positive_rate)
        result={}
        result['auc']=roc_auc
        result['ks']=ks
        return result

        # -------------------------------------------AUC-KS图形展示-----------------------------------------------------------------

    def plot_roc_ks(self,data,cut_points_result,score_card):
        data_score_proba = self.predict_score_proba(data,cut_points_result, score_card)
        false_positive_rate, recall, thresholds = roc_curve(data['y'], data_score_proba['proba'],drop_intermediate=False)
        roc_auc = auc(false_positive_rate, recall)

        plt.figure(figsize=(15, 15))

        #ROC曲线
        plt.subplot(211)
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out')

        #KS曲线
        plt.subplot(212)
        pre = sorted(data_score_proba['proba'], reverse=True)
        num = []
        for i in range(10):
            num.append((i) * int(len(pre) / 10))
        num.append(len(pre) - 1)

        df = pd.DataFrame()
        df['false_positive_rate'] = false_positive_rate
        df['recall'] = recall
        df['thresholds'] = thresholds
        data_ks = []
        for i in num:
            data_ks.append(list(df[df['thresholds'] == pre[i]].values[0]))
        data_ks = pd.DataFrame(data_ks)
        data_ks.columns = ['fpr', 'tpr', 'thresholds']
        ks = max(data_ks['tpr'] - data_ks['fpr'])
        plt.title('K-S曲线')
        plt.plot(np.array(range(len(num))), data_ks['tpr'])
        plt.plot(np.array(range(len(num))), data_ks['fpr'])
        plt.plot(np.array(range(len(num))), data_ks['tpr'] - data_ks['fpr'], label='K-S = %0.4f' % ks)
        plt.legend(loc='lower right')
        plt.xlim([0, 10])
        plt.ylim([0.0, 1.0])
        plt.ylabel('累计占比')
        plt.xlabel('分组编号')

        plt.show()

