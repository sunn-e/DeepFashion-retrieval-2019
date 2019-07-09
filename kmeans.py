# -*- coding:utf-8 -*-


from sklearn.cluster import KMeans
from retrieval import load_feat_db
from sklearn.externals import joblib
from config import DATASET_BASE, N_CLUSTERS
import os


if __name__ == '__main__':
    feats, _, labels = load_feat_db() #d_feats. c_feats
    #model = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_jobs=-1).fit(feats)
    model = KMeans(n_clusters=N_CLUSTERS, random_state=1, n_jobs=1).fit(feats)
    model_path = os.path.join(DATASET_BASE, r'models', r'kmeans.m')
    joblib.dump(model, model_path)
