"""train GMM Models
__author__ = Sohaib"""

import os
import pickle
import numpy as np
from sklearn import mixture
import argparse

Latent_Vector_Length = 640

def get_features(path):
    print ("Loading",path)
    data = pickle.load(open(path , mode='rb'))
    print (data.keys())
    print (data['h'].shape)
    return data


def train_GMM(data,num_labels,c):
    GMM = []
    x = data['h']
    y = data['label']
    print (x.shape)
    for i in range(num_labels):
        ind = np.where (y == i)
        X = x[ind]
        gmm = mixture.GaussianMixture(n_components=c,n_init=10,init_params='random').fit(X)
        GMM.append(gmm)
    return GMM

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-d', '--dataset', type=str, default='cifar', help='dataset cifar|imagenet|gtsrb')

    parser.add_argument('-i', '--data_dir', type=str, default='', help='Location for loading data data')
    parser.add_argument('-m', '--model_dir', type=str, default='', help='Classifier Model')
    args = parser.parse_args()

    args.data_dir = os.path.join( os.getcwd(), '..', f'data/{args.dataset}/Representation' )
    args.model_dir = "GMM_Models"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if args.dataset == 'imagenet':
        num_labels = 16
        components = 5
    elif args.dataset == 'cifar':
        num_labels = 10
        components = 8
    else:
        num_labels = 43
        components = 8

    train = get_features(f"{args.data_dir}/train")
    val = get_features(f"{args.data_dir}/test")

    GMM = train_GMM(train,num_labels,components)

    Sum = 0
    v_h = val['h']
    v_y = val['label']
    for i in range(v_h.shape[0]):
        sample = np.expand_dims(v_h[i],axis=0)
        Sum += GMM[v_y[i]].score_samples(sample)
    print ("GMM_Score",Sum/(v_h.shape[0]*1000))

    pkl_filename = f"{args.model_dir}/{args.dataset}_GMM"
    with open(pkl_filename, 'wb') as file:
            pickle.dump(GMM, file)
