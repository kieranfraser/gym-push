import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
import time
import os

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.ticker as ticker

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

import seaborn as sns
from matplotlib import pyplot as plt

class ClassificationBattery:
    
    def __init__(self, data, data_set_type, notif_cat, context_cat, context_scale, dir_path):
        sns.set(rc={'figure.figsize':(6,6)}, style='white', palette='pastel')
        self.data = data
        self.notif_cat = notif_cat
        self.context_cat = context_cat
        self.context_scale = context_scale
        self.notif_ohe = joblib.load(dir_path+'/data/user_1/notif_ohe.joblib')
        self.context_ohe = joblib.load(dir_path+'/data/user_1/context_ohe.joblib')
        self.context_scaler = joblib.load(dir_path+'/data/user_1/context_scaler.joblib')
        self.data_set_type = data_set_type
        self.dir_path = dir_path
        
        self.models = ['Adaboost'
                       'Random Forest', 
                       'Nearest Neighbors',
                       'Decision Tree',
                       'Naive Bayes']
        
        if data_set_type is 'train':
            self.data_time_period = '3months'
        elif: 
            self.data_time_period = '6months'
        else:
            self.data_time_period = '3months'
            
        
        self.results = []
                                          
    def encode(self):
        notif_enc = self.notif_ohe.transform(self.data[self.notif_cat]) 
        context_enc_cat = self.context_ohe.transform(self.data[self.context_cat])
        context_enc_scale = self.context_scaler.transform(self.data[self.context_scale]) 
        return np.concatenate([notif_enc, context_enc_cat, context_enc_scale], axis=1)
    
    def save_results(self):
        joblib.dump(self.results, self.dir_path+'/results/'+self.data_set_type+'/results.joblib')
        
        tmp = pd.DataFrame(self.results)
        ax = sns.catplot(x="metric", y="score", hue="model", data=tmp, height=6, kind="bar")
        ax.set(xlabel='Metric', ylabel='Score', title='Model Performance Predicting Notification Action')
        ax._legend.set_title('Model')
        plt.savefig(self.dir_path+'/results/'+self.data_set_type+'results.png', bbox_inches='tight')
        
        return self.results
        
    
    def execute(self):
        
        X = self.encode()
        y = self.data.action.values
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=888)
        
        for name in self.models:
            print('__Model__: '+name)
            
            model = joblib.load(self.dir_path+'/data/user_1/'+self.data_time_period+'_'+name+'.joblib')
            
            cv_results = cross_val_score(model, X, y, cv=sss, scoring='accuracy')
            self.results.append({'model':name, 'metric':'Accuracy', 'score':cv_results.mean()})
            cv_results = cross_val_score(model, X, y, cv=sss, scoring='precision')
            self.results.append({'model':name, 'metric':'Precision', 'score':cv_results.mean()})
            cv_results = cross_val_score(model, X, y, cv=sss, scoring='recall')
            self.results.append({'model':name, 'metric':'Recall', 'score':cv_results.mean()})
            cv_results = cross_val_score(model, X, y, cv=sss, scoring='f1')
            self.results.append({'model':name, 'metric':'F1', 'score':cv_results.mean()})
            
            
        return self.save_results()