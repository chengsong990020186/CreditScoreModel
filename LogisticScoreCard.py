# coding=utf-8
# Author: chengsong
# Time: 2019-01-15

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm # 在jupyter notebook使用
from numpy import inf

import logging
import matplotlib.pyplot as plt

plt.rc('font', family='SimHei', size=13)  # 使图形输出为中文

# logging说明
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
                 max_depth=None,  # 决策树的深度
                 max_leaf_nodes=4,  # 决策树的子节点数
                 min_samples_leaf=0.05,  # 分节点最小划分比例
                 base_score=600,  # 基础分
                 increase_score=50,  # 增加分（概率增加一份所增加的分数）
                 min_iv=0.1,  # 特征筛选（选取IV值大于0.1的值）
                 max_corr=0.6,  # 特征筛选（选取相关性小于0.6的值）
                 C=0.01,  # 特征筛选（L1正则化系数）
                 penalty='l1',  # 特征筛选（L1正则化）
                 round_num=2  # 所有变量保留两位有效数
                 ):

        # 参数选择
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.base_score = base_score
        self.increase_score = increase_score
        self.min_iv = min_iv
        self.max_corr = max_corr
        self.C = C
        self.penalty = penalty
        self.round_num = round_num

        # 保存变量结果
        self.col_type_iv = None  # 各变量类型以及IV值
        self.col_continuous_cut_points = None  # 连续变量的切分点，按小于等于，大于切分，空值单独归位一类，例如:['scorecashon', [-inf, 654.0, 733.0, 754.0, inf]]
        self.col_result = None  # 最终评分卡选择的变量
        self.score_card = None  # 评分卡

    def check_data_y(self, data):
        '''
        检查数据结构，数据预测变量为0,1，并以“y”命名
        '''

        if 'y' not in data.columns:
            logging.ERROR('未检测到"y"变量，请将预测变量命名改为"y"')

    # 读取原始数据
    # 划分训练集和测试集
    def get_data_train_test(self, data, test_size=0.25, random_state=1234):
        self.check_data_y(data)
        x_train, x_test, y_train, y_test = train_test_split(data[[col for col in data.columns if col != 'y']],
                                                            data['y'], test_size=test_size,
                                                            random_state=random_state)  # 随机选择25%作为测试集，剩余作为训练集
        data_train = x_train.reset_index()
        del data_train['index']
        data_train['y'] = y_train.reset_index()['y']
        data_test = x_test.reset_index()
        del data_test['index']
        data_test['y'] = y_test.reset_index()['y']
        return data_train, data_test

    # 划分离散和连续变量（连续变量：int64,float64；离散变量：其他）
    def get_col_discrete_continue(self, data):
        logging.info('连续和离散变量划分中。。。')
        col_all = data.columns
        col_all_type = data.dtypes
        col_type = []
        for i in range(len(col_all)):
            if col_all[i] != 'y':
                if str(col_all_type[i]) in ('int64', 'float64'):
                    col_type.append([col_all[i], 'continuous'])
                else:
                    col_type.append([col_all[i], 'discrete'])
        logging.info('连续和离散变量划分完成！')
        return col_type

    # ----------------------------------------分箱（决策树）--------------------------------------------------------------------

    def get_descison_tree_cut_point(self, data, col, max_depth=None, max_leaf_nodes=4, min_samples_leaf=0.05):
        data_notnull = data[[col, 'y']][data[col].notnull()]  # 删除空值
        cut_point = []
        if len(np.unique(data_notnull[col])) > 1:
            x = data_notnull[col].values.reshape(-1, 1)
            y = data_notnull['y'].values

            clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                         max_depth=max_depth,  # 树的深度
                                         max_leaf_nodes=max_leaf_nodes,  # 最大叶子节点数
                                         min_samples_leaf=min_samples_leaf)  # 叶子节点样本数量最小占比
            clf.fit(x, y)  # 训练决策树

            threshold = np.unique(clf.tree_.threshold)
            x_num = np.unique(x)

            for i in threshold:
                if i != -2:
                    point = np.round(max(x_num[x_num < i]), 2)  # 取切分点左边的数
                    cut_point.extend([point])
            cut_point = [float(str(i)) for i in cut_point]
            cut_point = [-inf] + cut_point + [inf]
        return cut_point

    # ------------------------------------------------------------------------------------------------------------

    # 根据切分点切分变量数据
    def get_cut_result(self, data, col_continuous_cut_points):
        logging.info('根据cut离散化连续变量进行中。。。')
        cols = [i for i in data.columns if i not in [i[0] for i in col_continuous_cut_points]]
        data_cut_result = data[cols].copy()
        for col, cut_points in tqdm(col_continuous_cut_points):
            data_cut_result[col] = pd.cut(data[col], cut_points).astype("str")

        data_cut_result = data_cut_result.fillna('null')
        data_cut_result.replace('nan', 'null', inplace=True)
        logging.info('根据cut离散化连续变量完成！')
        return data_cut_result

    # 获取按切分点的统计数据
    def get_woe_iv(self, data_discrete, col):
        result = data_discrete.groupby(col)['y'].agg([('1_num', lambda y: (y == 1).sum()),
                                                      ('0_num', lambda y: (y == 0).sum()),
                                                      ('total_num', 'count')]).reset_index()
        result['1_pct'] = result['1_num'] / result['1_num'].sum()
        result['0_pct'] = result['0_num'] / result['0_num'].sum()
        result['total_pct'] = result['total_num'] / result['total_num'].sum()
        result['1_rate'] = result['1_num'] / result['total_num']
        result['woe'] = np.log(result['1_pct'] / result['0_pct'])  # WOE
        result['iv'] = (result['1_pct'] - result['0_pct']) * result['woe']  # IV
        result['total_iv'] = result['iv'].sum()
        result.replace([-inf, inf], [0, 0], inplace=True)
        result = result.rename(columns={col: "cut_points"})
        return result

    # 批量获取变量IV值
    def get_iv(self, data):
        logging.info('IV值计算中。。。')
        col_iv = []
        for col in tqdm([i for i in data.columns if i != 'y']):
            col_woe_iv = self.get_woe_iv(data, col)
            col_iv.append([col, col_woe_iv['iv'].sum()])
        logging.info('IV值计算完成！')
        return col_iv

    # 数据转换为woe
    def get_data_woe(self, data_discrete):
        logging.info('WOE转换中。。。')
        data_woe = pd.DataFrame()
        for col in tqdm([i for i in data_discrete.columns if i != 'y']):
            col_woe_iv = self.get_woe_iv(data_discrete, col)
            data_woe[col] = data_discrete[col].replace(list(col_woe_iv['cut_points']), list(col_woe_iv['woe']))
        data_woe['y'] = data_discrete['y']
        logging.info('WOE转换完成！')
        return data_woe

    # 通过IV值和相关性以及逻辑回归L1选择变量（默认参数提出IV值>=0.1，相关系数>0.6,l1正则筛选的变量）
    def get_iv_corr_logistic_l1_col(self, data_woe, col_iv, min_iv=0.1, max_corr=0.6, C=0.01, penalty='l1'):
        logging.info('根据IV值大于 %s 且 相关性小于 %s ，以及l1正则选取变量进行中。。。' % (min_iv, max_corr))
        col_filter = [col for col, iv in col_iv if iv > 0.1]
        col_iv_filter = [[col, iv] for col, iv in col_iv if iv > 0.1]
        data_woe_corr = data_woe[col_filter].corr()
        data_woe_corr_list = data_woe_corr.values.reshape(-1, 1)
        col_iv_result = []
        for col1, iv1 in col_iv_filter:
            for col2, iv2 in col_iv_filter:
                col_iv_result.append([col1, col2, iv1, iv2, iv1 - iv2])

        data_woe_corr_iv = pd.DataFrame(col_iv_result, columns=['col1', 'col2', 'iv1', 'iv2', 'iv1_iv2'])
        data_woe_corr_iv['corr'] = data_woe_corr_list
        # 剔除相关性较大，而IV值较低的变量
        col_delete = data_woe_corr_iv['col1'][(data_woe_corr_iv['corr'] < 1) & (data_woe_corr_iv['corr'] > 0.6) & (
                data_woe_corr_iv['iv1_iv2'] < 0)].unique()
        col_filter_result = [col for col in col_filter if col not in (col_delete)]

        # L1正则化筛选
        lr = linear_model.LogisticRegression(C=C, penalty=penalty).fit(data_woe[col_filter_result], data_woe['y'])
        col_result = [col_filter_result[i] for i in range(len(col_filter_result)) if lr.coef_[0][i] != 0]
        logging.info('变量选取完成，总共 %s 个变量，最终筛选出 %s 个变量' % (len(data_woe.columns) - 1, len(col_result)))
        return col_result

    def get_logistic_socre_card(self, data, col_continuous_cut_points, increase_score=50, base_score=600):
        logging.info('评分卡制作中。。。')
        col_types = self.get_col_discrete_continue(data)
        col_result = [i for i in data.columns if i != 'y']
        data_discrete = self.get_cut_result(data, col_continuous_cut_points)  # 按切分点划分数据，得到全部的离散数据
        data_woe = self.get_data_woe(data_discrete)  # 数据woe化
        # 评分卡制作
        lr = linear_model.LogisticRegression(C=1, penalty='l2')
        lr.fit(data_woe[col_result], data_woe['y'])
        b = -increase_score / np.log(2)
        a = base_score - lr.intercept_[0] * b

        score_card = pd.DataFrame()
        for col in col_result:
            col_cut_point_woe = self.get_woe_iv(data_discrete, col)
            col_cut_point_woe['col'] = col
            score_card = pd.concat([score_card, col_cut_point_woe])

        col_coef = pd.DataFrame(col_result, lr.coef_[0]).reset_index()
        col_coef.columns = ['col_coef', 'col']
        score_card['lr_intercept'] = lr.intercept_[0]
        score_card = pd.merge(score_card, col_coef, on=['col'], how='left')
        score_card['score'] = score_card['woe'] * score_card['col_coef'] * b
        score_card = pd.merge(score_card, pd.DataFrame(col_types, columns=['col', 'type']), on='col', how='left')
        score_card = pd.merge(score_card, pd.DataFrame(col_continuous_cut_points, columns=['col', 'cuts']), on='col',how='left')

        # 切分点排序
        data_cut_points_id = pd.DataFrame()
        for col, cut_point in col_continuous_cut_points:
            result = pd.DataFrame()
            result['cut_points'] = pd.cut(data[col], cut_point).astype('str').unique()
            result['cut_points_id'] = pd.cut(data[col], cut_point).unique()._codes
            result['cut_points'].replace('nan', 'null', inplace=True)
            result['col'] = col
            data_cut_points_id = pd.concat([data_cut_points_id, result])
        score_card = pd.merge(score_card, data_cut_points_id, on=['col', 'cut_points'], how='left').sort_values(
            ['col', 'cut_points_id', 'cut_points'])

        score_card = score_card[
            ['col', 'type', 'cuts', 'cut_points', '1_num', '0_num', 'total_num', '1_pct', '0_pct', 'total_pct',
             '1_rate', 'woe', 'iv', 'total_iv', 'col_coef', 'lr_intercept', 'score']].reset_index(drop=True)
        # score_card = score_card[
        #     ['变量名', '变量类型', '切分点', '切分分组', 'y为1的数量', 'y为0的数量', '总数', 'y为1的数量占比', 'y为0的数量占比', '总数占比',
        #      'y为1占总数比例', 'woe', '各分组iv', '变量iv值', 'logistic参数col_coef', 'logistic参数lr_intercept', '分组分数']].reset_index(drop=True)
        logging.info('评分卡制作完成！')
        return score_card

    def fit(self, data):
        data = data.round(self.round_num)  # 保留两位小数
        logging.info('任务开始。。。')
        self.check_data_y(data)

        # -----------------------------------------变量筛选--------------------------------------------------------------
        # 划分离散和连续变量
        col_types = self.get_col_discrete_continue(data)

        col_continuous_cut_points = []
        logging.info('连续变量最优分组进行中。。。')
        for col, col_type in tqdm(col_types):
            if col_type == 'continuous':
                point = self.get_descison_tree_cut_point(data[[col, 'y']], col, self.max_depth, self.max_leaf_nodes,
                                                         self.min_samples_leaf)
                if point:
                    col_continuous_cut_points.append([col, point])
        #     else:
        #         col_cut_points.append([col,'discrete',None])
        logging.info('连续变量最优分组完成！')

        data_discrete = self.get_cut_result(data, col_continuous_cut_points)  # 按切分点划分数据，得到全部的离散数据

        col_iv = self.get_iv(data_discrete)  # 各变量IV值

        data_woe = self.get_data_woe(data_discrete)  # 数据woe化

        col_result = self.get_iv_corr_logistic_l1_col(data_woe, col_iv, min_iv=0.1, max_corr=0.6, C=0.05,penalty='l1')  # 变量筛选
        # -----------------------------------------评分卡制作--------------------------------------------------------------

        col_result_continuous_cut_points = [col for col in col_continuous_cut_points if col[0] in col_result]
        score_card=self.get_logistic_socre_card(data[col_result + ['y']], col_result_continuous_cut_points, increase_score=self.increase_score, base_score=self.increase_score)

        # -----------------------------------------保存结果--------------------------------------------------------------

        col_type_iv = pd.merge(pd.DataFrame(col_types, columns=['col', 'type']),pd.DataFrame(col_iv, columns=['col', 'iv']), on='col', how='left')

        self.col_type_iv = col_type_iv  # 计算连续变量离散化后的IV值
        self.col_continuous_cut_points = col_continuous_cut_points  # 连续变量的切分点，按小于等于，大于切分，空值单独归位一类，例如:['scorecashon', [-inf, 654.0, 733.0, 754.0, inf]]
        self.col_result = col_result  # 最终评分卡选择的变量
        self.score_card = score_card  # 评分卡结果

        logging.info('任务完成！')

    def predict_score_proba(self, data, score_card, increase_score=50, base_score=600):
        logging.info('预测用户分数中。。。')
        b = -increase_score / np.log(2)
        a = base_score - score_card['lr_intercept'][0] * b
        col_result = score_card['col'].unique().tolist() + ['y']
        col_continuous_cut_points = score_card[['col', 'cuts']][score_card['type'] == 'continuous'].drop_duplicates('col').values.tolist()
        data_discrete = self.get_cut_result(data[col_result], col_continuous_cut_points)
        data_score_proba = pd.DataFrame()
        for col in score_card['col'].unique():
            col_score = col + 'score'
            cut_points = score_card['cut_points'][score_card['col'] == col].tolist()
            score = score_card['score'][score_card['col'] == col].tolist()
            data_score_proba[col_score] = data_discrete[col].replace(cut_points, score)
        data_score_proba['score'] = data_score_proba.sum(axis=1)+ score_card['lr_intercept'][0] * b + a
        data_score_proba['proba'] = 1 - 1 / (1 + np.e ** ((data_score_proba['score'] - a) / b))
        return data_score_proba

    def score(self, data, score_card):
        data_score_proba = self.predict_score_proba(data, score_card)
        false_positive_rate, recall, thresholds = roc_curve(data['y'], data_score_proba['proba'])
        roc_auc = auc(false_positive_rate, recall)
        ks = max(recall - false_positive_rate)
        result = {}
        result['auc'] = roc_auc
        result['ks'] = ks
        return result

    def plot_roc_ks(self, data, score_card):
        data_score_proba = self.predict_score_proba(data, score_card)
        false_positive_rate, recall, thresholds = roc_curve(data['y'], data_score_proba['proba'],drop_intermediate=False)
        roc_auc = auc(false_positive_rate, recall)
        plt.figure(figsize=(20, 10))

        # ROC曲线
        plt.subplot(121)
        plt.title('ROC')
        plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('false_positive_rate')
        plt.ylabel('recall')

        # KS曲线
        plt.subplot(122)
        pre = sorted(data_score_proba['proba'], reverse=True)
        num = [(i) * int(len(pre) / 10) for i in range(10)]
        num = num + [(len(pre) - 1)]
        ks_thresholds = [max(thresholds[thresholds <= pre[i]]) for i in num]
        data_ks = pd.DataFrame([false_positive_rate, recall, thresholds]).T
        data_ks.columns = ['fpr', 'tpr', 'thresholds']
        data_ks = pd.merge(data_ks, pd.DataFrame(ks_thresholds, columns=['thresholds']), on='thresholds', how='inner')
        ks = max(recall - false_positive_rate)
        plt.title('KS')
        plt.plot(np.array(range(len(num))), data_ks['tpr'])
        plt.plot(np.array(range(len(num))), data_ks['fpr'])
        plt.plot(np.array(range(len(num))), data_ks['tpr'] - data_ks['fpr'], label='K-S = %0.4f' % ks)
        plt.legend(loc='lower right')
        plt.xlim([0, 10])
        plt.ylim([0.0, 1.0])
        plt.xlabel('label')
        plt.show()

    def plot_col_woe_iv(self, data, col, cut_point=None, return_data=True):
        data_cut_result = pd.DataFrame()
        if cut_point:
            cut_point = cut_point
        else:
            cut_point = self.get_descison_tree_cut_point(data, col)
        df = pd.DataFrame()
        df['cut_points'] = pd.cut(data[col], cut_point).astype('str').unique()
        df['cut_points_id'] = pd.cut(data[col], cut_point).unique()._codes
        df['cut_points'].replace('nan', 'null', inplace=True)
        data_cut_result[col] = pd.cut(data[col], cut_point).astype('str')
        data_cut_result['y'] = data['y']
        data_cut_result.replace('nan', 'null', inplace=True)
        col_woe_iv = self.get_woe_iv(data_cut_result, col)
        col_woe_iv = pd.merge(df, col_woe_iv, on='cut_points').sort_values('cut_points_id')
        x = range(len(col_woe_iv))
        y1 = col_woe_iv['total_pct']
        y2 = col_woe_iv['1_rate']
        plt.bar(x, y1, 0.5)
        plt.plot(x, y2, 'r', label='IV = %0.2f' % col_woe_iv['iv'].sum())
        plt.title('各分组占比和逾期率')
        plt.ylim([0, 1])
        plt.xlabel('分组')
        plt.ylabel('百分比')
        plt.xticks(x, col_woe_iv['cut_points'])
        plt.legend(loc='upper right')
        plt.show()
        if return_data == True:
            return col_woe_iv
