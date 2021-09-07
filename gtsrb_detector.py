
"""
Detection using direct comparison between input image and generated view G4.
Though this simple detection scheme is only effective for simpler dataset like GTSRB


__author__ = Sohaib Kiani
August 2021
"""

import pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
import decimal
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += decimal.Decimal(step)



def load_test_data(method):
    path_clean = 'data/Generated/gtest_c'
    path_adv = f'data/Generated/{method}'

    data = pickle.load(open(path_clean , mode='rb'))
    print (data.keys())
    x = data['data']
    x = ((x+1)/2.0)*255
    y = data['label']

    data = pickle.load(open(path_adv , mode='rb'))
    print (data.keys())
    x_adv = data['data']
    #print ("Range of data should be -1-1 and actual is: ",str(np.min(data['data']))+" "+str(np.max(data['data'])))
    y_adv = data['alabel']
    x_adv = ((x_adv+1)/2.0)*255
    return x,y,x_adv,y_adv

if __name__ == "__main__":
     threshold = 32
     for method in attacks:
         data_clean,y1,data_adv,y2 = load_test_data(method)
         auc_plot = []
         th_auc = float_range(0,100,0.1)
         for threshold in th_auc:
             dist = []
             label = []
             y_pred = []
             for i in range(0,500,5):
                 ref = data_clean[i].reshape(1,-1)
                 comp = data_clean[(i) + 4].reshape(1,-1)
                 d = distance.euclidean(ref,comp)/100
                 dist.append(d)
                 if d < threshold:
                     y_pred.append(0)
                 else:
                     y_pred.append(1)
                 label.append(0)
                 ref = data_adv[i].reshape(1,-1)
                 comp = data_adv[(i) + 4].reshape(1,-1)
                 d = distance.euclidean(ref,comp)/100
                 if d < threshold:
                     y_pred.append(0)
                 else:
                     y_pred.append(1)
                 label.append(1)
             tn, fp, fn, tp = metrics.confusion_matrix(label, y_pred).ravel()
             tpr = tp/(tp+fn)
             fpr = fp/(tn+fp)
             auc_plot.append((tpr,fpr))
         auc = metrics.roc_auc_score(label, y_pred)
         print (f"AUC Score for {method}: {auc}")


        #print ("Next Sample")
