
"""
__author__ = Sohaib Kiani
August 2021

Training and evaluation and adversarial Detector
30% of the samples are used for Training and remaining for Testing

Output AUC Score for Various Attack Method.
Detection Rate can be obtained by setting threshold to fix value that can achieve TNR: 95%
ROC plot code has also been provided
"""


from sklearn.metrics import classification_report
from sklearn import metrics
import pandas as pd
import decimal
import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import matplotlib.pyplot as plt
#plt.style.use('seaborn')
class Classifier():
    def __init__(self,dataset):
        if dataset == 'cifar':
            self.supervised = RandomForestClassifier(max_depth = 7,class_weight = {-1:0.5,1:4.2})
            self.Novelty = IsolationForest(max_samples=400,random_state=42,max_features=2,bootstrap = True,n_estimators = 20,contamination = 0.001)
        if dataset == 'imagenet':
            self.supervised = RandomForestClassifier(max_depth = 7,class_weight = {-1:0.5,1:2.2})
            self.Novelty = IsolationForest(max_samples=400,random_state=42,max_features=2,bootstrap = True,n_estimators = 20,contamination = .001)
        if dataset == 'gtsrb':
            self.supervised = RandomForestClassifier(max_depth = 7,class_weight = {-1:0.5,1:4.2})
            self.Novelty = IsolationForest(max_samples=400,random_state=42,max_features=2,bootstrap = True,n_estimators = 20,contamination = 0.001)
    def train_supervised(self,X_train,y_train):
        self.supervised.fit(X_train,y_train)
    def train_Novelty(self,X_train):
        self.Novelty.fit(X_train)


    def predict(self,X_test,th):
        level1 = self.supervised.predict_proba(X_test)
        level2 = self.Novelty.score_samples(X_test)
        pred_out = level2.copy()
        pred_out[:] = -1
        count = 0
        for i in range(level1.shape[0]):
            if level1[i,1] + level2[i] > th:
                pred_out[i] = 1
        return pred_out


def train_supervised(Detector,feature_list,samples):

    df_x = pd.read_pickle(f'{args.data_dir}/g_test_c')
    df_a1 = pd.read_pickle(f'{args.data_dir}/g_deepFool')
    df_a2 = pd.read_pickle(f'{args.data_dir}/g_pgd_E4')

    X  = df_x.iloc[:samples]
    X = X.append(df_a1.iloc[:int(samples/2)], ignore_index = True)
    X = X.append(df_a2.iloc[:int(samples/2)], ignore_index = True)
    print (X.Label.nunique())
    print (X.head())
    X.drop(feature_list,axis=1,inplace=True)
    y = X.pop('Label')

    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.01, random_state=2)
    Detector.train_supervised(X_train,y_train)
    print (sorted(Detector.supervised.feature_importances_))

def float_range(start, stop, step):
    temp = []
    while start < stop:
        temp.append(float(start))
        start = decimal.Decimal(start) + decimal.Decimal(step)
    return temp

def Find_Likelihood(df):
    print (df.head())
    x = df.Px.sum()
    print ("Px Likelihood",x/df.shape[0])
    df['Ph'] = df.apply(lambda elem : (elem['ph_0'] * -1000)/(np.log(2.)*32*32*3), axis = 1)
    print (df.head())

def ROC_plot(Detector,samples):
    map = {'g_WB':'WhiteBox','g_deepFool':'DeepFool','g_CW':'C&W','g_mim_E4':'MIM','g_fgsm_E4':'FGSM','g_pgd_E4':'PGD-4'}
    l_color = ['orange','y','g','b','b','grey']
    attack_method = ['g_deepFool','g_pgd_E4','g_fgsm_E4','g_mim_E4','g_CW','g_WB']
    count = 0
    auc_avg = 0
    for method in attack_method:
        X = pd.read_pickle(f'{args.data_dir}/g_test_c').iloc[train_samples:]
        print (X.shape)
        df_a = pd.read_pickle(f'{args.data_dir}/{method}').iloc[train_samples:]
        print (df_a.shape)
        X = X.append(df_a, ignore_index = True)
        X.drop(feature_list,inplace = True, axis=1)
        y_test =  X.pop('Label')
        auc_plot = []
        th_range = float_range(-1,1,'0.1')
        for th in th_range:
            y_det =  Detector.predict(X,th)
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_det).ravel()
            tpr = tp/(tp+fn)
            fpr = fp/(tn+fp)
            auc_plot.append((tpr,fpr))
        print (f"AUC Score {method}")
        tpr_temp = [a_tuple[0] for a_tuple in auc_plot]
        fpr_temp = [a_tuple[1] for a_tuple in auc_plot]
        auc_score = "{:.3f}".format(metrics.auc(fpr_temp, tpr_temp))
        print (f"AUC Score {method}: {auc_score}")
        plt.plot(fpr_temp, tpr_temp, linestyle='-',linewidth='1',color=l_color[count], label=f"{map[method]}, AUC = {auc_score}")
        count += 1
    plt.plot([0,1],[0,1],'r--')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid()
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc='lower right',fontsize="x-small",frameon=True)
    plt.axes().set_aspect('equal')
    plt.savefig(f'{args.dataset}_ROC',dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-d', '--dataset', type=str, default='imagenet', help='dataset cifar|imagenet|gtsrb')
    parser.add_argument('-i', '--data_dir', type=str, default='', help='Location Final labelled feature files')
    args = parser.parse_args()
    args.data_dir = f'data/{args.dataset}/Final_Descriptors'

    Detector = Classifier(args.dataset)
    df_x = pd.read_pickle(f'{args.data_dir}/g_test_c')
    y = df_x.pop('Label')
    train_samples = int(df_x.shape[0] * 0.3)
    Detector.train_Novelty(df_x.iloc[:train_samples])
    feature_list = []
    train_supervised(Detector,feature_list,train_samples)

    attack_method = ['g_deepFool','g_CW','g_pgd_E4','g_pgd_E8','g_pgd_E16','g_fgsm_E4','g_fgsm_E8','g_fgsm_E16','g_mim_E4','g_mim_E8','g_mim_E16']
    count = 0
    auc_avg = 0
    for method in attack_method:
        X = pd.read_pickle(f'{args.data_dir}/g_test_c').iloc[train_samples:]
        print (X.shape)
        df_a = pd.read_pickle(f'{args.data_dir}/{method}').iloc[train_samples:]
        print (df_a.shape)
        X = X.append(df_a, ignore_index = True)
        X.drop(feature_list,inplace = True, axis=1)
        y_test =  X.pop('Label')
        th_range = float_range(-1,1,'0.05')
        auc_plot = []
        for th in th_range:
            y_det =  Detector.predict(X,th)
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_det).ravel()
            tpr = tp/(tp+fn)
            fpr = fp/(tn+fp)
            auc_plot.append((tpr,fpr))
        print (f"AUC Score {method}")
        tpr_temp = [a_tuple[0] for a_tuple in auc_plot]
        fpr_temp = [a_tuple[1] for a_tuple in auc_plot]
        #print (fpr_temp)
        auc_score = "{:.3f}".format(metrics.auc(fpr_temp, tpr_temp))
        print (f"AUC Score {method}: {auc_score}")
    #ROC_plot(Detector,train_samples)
