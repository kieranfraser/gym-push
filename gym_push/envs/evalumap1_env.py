import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pandas as pd
import numpy as np
from json_tricks import dumps

from random import randrange, choice
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.ticker as ticker

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

import seaborn as sns
from matplotlib import pyplot as plt

class EvalUMAP1(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        print('EvalUMAP1-v0 environment')
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        
        # ------------ load data ------------------ #
        # ------------ train as default ------------------ #
        self.data = pd.read_csv(self.dir_path+'/data/user_1/train_set.csv' )
        self.clf = joblib.load(self.dir_path+'/data/user_1/3months_Adaboost.joblib')
        self.notif_ohe = joblib.load(self.dir_path+'/data/user_1/notif_ohe.joblib')
        self.context_ohe = joblib.load(self.dir_path+'/data/user_1/context_ohe.joblib')
        self.context_scaler = joblib.load(self.dir_path+'/data/user_1/context_scaler.joblib')
        
        self.notif_cat = ['appPackage', 'category', 'ledARGB', 'priority', 'vibrate', 'visibility', 'subject', 'enticement', 'sentiment']
        self.context_cat = ['timeAppLastUsed', 'timeOfDay', 'dayOfWeek']
        self.context_scale = ['unlockCount_prev2', 'uniqueAppsLaunched_prev2', 'dayOfMonth']
        
        self.contexts = self.data[(self.context_cat+self.context_scale)]
        self.notifications = self.data[self.notif_cat]
        self.action_space = DataSpace(self.notifications)
        self.observation_space = DataSpace(self.contexts)
        
        # Will always be based on train data, as test is unseen
        self.max_diversity = self.data[self.notif_cat].nunique().sum()
        
        # ------------- Create the engagements using the classifier ------- #
        self.engagements = self.predictEngagements(self.notifications, self.contexts)
        self.engagements = self.predictEngagements(self.notifications, self.contexts)
        
        # These are now different  - the action space is made up of combinations of possible Notifications
        # the observation space is made up of the possible contexts
        # best use label encoders
        
        self.epoch = 0
        self.openedNotifications = []
        self.openedActions = []
        self.openedContexts = []
        self.dismissedNotifications = []
        self.dismissedActions = []
        self.dismissedContexts = []
        self.correctlyOpened = 0
        self.correctlyDismissed = 0
        self.test = False
        
        # ------------ evaluation variables ------------------ #
        sns.set(rc={'figure.figsize':(6,6)}, style='white', palette='pastel')
        self.models = ['Adaboost',
                       # 'Random Forest', 
                       # 'Nearest Neighbors',
                       'Decision Tree',
                       'Naive Bayes']
        plt.ioff() # turn off interactive mode, so will only show on call
        
    def step(self, action):
        reward = self.calculate_reward(action)
        done = False
        self.epoch = self.epoch + 1
        if self.epoch == (len(self.notifications)):
            done = True
            observation = {}
        else:
            observation = {**self.notifications.iloc[self.epoch].to_dict(), **self.contexts.iloc[self.epoch].to_dict()}
        info = {}
        return observation, reward, done, info
    
    def reset(self):
        
        # ------------ load data ------------------ #
        # ------------ train as default ------------------ #
        self.data = pd.read_csv(self.dir_path+'/data/user_1/train_set.csv' )
        self.clf = joblib.load(self.dir_path+'/data/user_1/3months_Adaboost.joblib')
        self.notif_ohe = joblib.load(self.dir_path+'/data/user_1/notif_ohe.joblib')
        self.context_ohe = joblib.load(self.dir_path+'/data/user_1/context_ohe.joblib')
        self.context_scaler = joblib.load(self.dir_path+'/data/user_1/context_scaler.joblib')
        
        self.contexts = self.data[(self.context_cat+self.context_scale)]
        self.notifications = self.data[self.notif_cat]
        self.action_space = DataSpace(self.notifications)
        self.observation_space = DataSpace(self.contexts)
        
        # ------------- Create the engagements using the classifier ------- #
        self.engagements = self.predictEngagements(self.notifications, self.contexts)
        
        # These are now different  - the action space is made up of combinations of possible Notifications
        # the observation space is made up of the possible contexts
        # best use label encoders
        
        self.epoch = 0
        self.openedNotifications = []
        self.openedActions = []
        self.openedContexts = []
        self.dismissedNotifications = []
        self.dismissedActions = []
        self.dismissedContexts = []
        self.correctlyOpened = 0
        self.correctlyDismissed = 0
    
    def close(self):
        print('To do.. close')
        
    # ----------- Metric methods ----------#
    
    def autopct_generator(self, ignored):
        def inner_autopct(pct):
            if round(pct, 2)==ignored:
                return ''
            else:
                return ('%.2f' % pct)+'%'
        return inner_autopct
    
    def save_results(self, ctr_results, diversity_score, enticement_score, data_set_type):
        
        tmp = pd.DataFrame(ctr_results)
        ax = sns.catplot(x="model", y="ctr_score", data=tmp, height=6, kind="bar")
        ax.set(xlabel='Model', ylabel='CTR (%)', title="CTR performance of notifications")
        plt.savefig(self.dir_path+'/results/'+data_set_type+'/ctr_results.png', bbox_inches='tight')
        plt.clf()
        
        adaboost_score = ctr_results[0]['ctr_score']
        # AdaBoost
        if round(adaboost_score, 2) == 50.00:
            adaboost_score = 50.01
        plt.rcParams.update({'font.size': 22})
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))
        size=[round(adaboost_score,2), round(np.absolute(100-adaboost_score),2)]
        my_circle=plt.Circle( (0,0), 0.7, color='white')
        plt.pie(size, autopct=self.autopct_generator(round(np.absolute(100-adaboost_score),2)),
                pctdistance=0.01, colors=['aquamarine','peachpuff'])
        ax.set_title("Click-Through-Rate Score", fontsize= 22)
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        plt.savefig(self.dir_path+'/results/'+data_set_type+'/ctr_pie.png', bbox_inches='tight')
        
        # enticement
        if round(enticement_score, 2) == 50.00:
            enticement_score = 49.99
        plt.rcParams.update({'font.size': 22})
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))
        size=[round(enticement_score,2), round(np.absolute(100-enticement_score),2)]
        my_circle=plt.Circle( (0,0), 0.7, color='white')
        plt.pie(size, autopct=self.autopct_generator(round(np.absolute(100-enticement_score),2)),
                pctdistance=0.01, colors=['peachpuff','aquamarine'])
        ax.set_title("Enticement Score", fontsize= 22)
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        plt.savefig(self.dir_path+'/results/'+data_set_type+'/enticement_pie.png', bbox_inches='tight')
        
        # diversity
        if round(diversity_score, 2) == 50.00:
            diversity_score = 50.01
        plt.rcParams.update({'font.size': 22})
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))
        size=[round(diversity_score,2), round(np.absolute(100-diversity_score),2)]
        my_circle=plt.Circle( (0,0), 0.7, color='white')
        plt.pie(size, autopct=self.autopct_generator(round(np.absolute(100-diversity_score),2)),
                pctdistance=0.01, colors=['aquamarine','peachpuff'])
        ax.set_title("Diversity Score", fontsize= 22)
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        plt.savefig(self.dir_path+'/results/'+data_set_type+'/diversity_pie.png', bbox_inches='tight')
        
        plt.show()
        ctr_results.append({'metric': 'diversity_score', 'score': diversity_score})
        ctr_results.append({'metric': 'enticement_score', 'score': enticement_score})
        joblib.dump(ctr_results, self.dir_path+'/results/'+data_set_type+'/results.joblib')
        
    def enticement_to_value(self, val):
        if val == 'high':
            return 3
        elif val == 'moderate':
            return 2
        else:
            return 1
    
    def execute_evaluation(self, notifications, contexts, test):
        if test:
            data_time_period = '6months'
        else:
            data_time_period = '3months'
            
        # Set the max diversity equal to the sum of possible values
        # per feature of original notification data
        diversity_score = (notifications.nunique().sum()/self.max_diversity)*100
        
        a = len(contexts)
        b = ((a*3)-a)
        enticement_score = ((notifications.enticement.apply(self.enticement_to_value).sum() - a)/b)*100
            
        ctr_results = []
        X = self.encode(notifications, contexts)
        y = self.engagements
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=888)
        
        for name in self.models:
            print('__Model__: '+name)
            
            model = joblib.load(self.dir_path+'/data/user_1/'+data_time_period+'_'+name+'.joblib')
            predictions = model.predict(X)
            
            
            # self.results.append({'model':name, 'metric':'CTR', ((np.sum(predictions)/len(predictions))*100)})
            ctr_results.append({'model': name, 'ctr_score': ((np.sum(predictions)/len(predictions))*100)})
            
            '''cv_results = cross_val_score(model, X, y, cv=sss, scoring='accuracy')
            self.results.append({'model':name, 'metric':'Accuracy', 'score':cv_results.mean()})
            cv_results = cross_val_score(model, X, y, cv=sss, scoring='precision')
            self.results.append({'model':name, 'metric':'Precision', 'score':cv_results.mean()})
            cv_results = cross_val_score(model, X, y, cv=sss, scoring='recall')
            self.results.append({'model':name, 'metric':'Recall', 'score':cv_results.mean()})
            cv_results = cross_val_score(model, X, y, cv=sss, scoring='f1')
            self.results.append({'model':name, 'metric':'F1', 'score':cv_results.mean()})'''
            
        self.save_results(ctr_results, diversity_score, enticement_score, 'test' if test else 'train')
        return ctr_results
        
    def calculate_reward(self, action):
        ground_truth = self.engagements
        self.update_metrics(ground_truth, action)
        if ground_truth == action:
            return 1
        else:
            return -1 
    
    def update_metrics(self, ground_truth, action):
        if action == 1:
            self.openedActions.append(action)
            self.openedNotifications.append(self.notifications.iloc[self.epoch].to_dict())
            self.openedContexts.append(self.contexts.iloc[self.epoch].to_dict())
            if ground_truth == 1:
                self.correctlyOpened = self.correctlyOpened + 1
        if action == 0:
            self.dismissedActions.append(action)
            self.dismissedNotifications.append(self.notifications.iloc[self.epoch].to_dict())
            self.dismissedContexts.append(self.contexts.iloc[self.epoch].to_dict())
            if ground_truth == 0:
                self.correctlyDismissed = self.correctlyDismissed + 1
    
    # ----------- Simulated user methods ----------#
    
    def encode(self, notifications, contexts):
        notif_enc = self.notif_ohe.transform(notifications) 
        context_enc_cat = self.context_ohe.transform(contexts[self.context_cat])
        context_enc_scale = self.context_scaler.transform(contexts[self.context_scale]) 
        return np.concatenate([notif_enc, context_enc_cat, context_enc_scale], axis=1)
    
    def predictEngagements(self, notifications, contexts):
        enc = self.encode(notifications, contexts)
        return self.clf.predict(enc)
    
    # ----------- EvalUMAP methods ----------#
    
    def request_data(self, test=False):
        self.reset()
        
        if test:
            self.test = True
            self.data = pd.read_csv(self.dir_path+'/data/user_1/test_set.csv' )
            self.clf = joblib.load(self.dir_path+'/data/user_1/6months_Adaboost.joblib')

            self.contexts = self.data[(self.context_cat+self.context_scale)]
            self.notifications = self.data[self.notif_cat]
            self.action_space = {'info':{}}
            self.observation_space = {'info':{}}

            # ------------- Create the engagements using the classifier ------- #
            self.engagements = self.predictEngagements(self.notifications, self.contexts)
            self.engagements = pd.DataFrame(self.engagements, columns=['action'])
        else:
            self.test = False
            self.data = pd.read_csv(self.dir_path+'/data/user_1/train_set.csv' )
            self.clf = joblib.load(self.dir_path+'/data/user_1/3months_Adaboost.joblib')

            self.contexts = self.data[(self.context_cat+self.context_scale)]
            self.notifications = self.data[self.notif_cat]
            self.action_space = DataSpace(self.notifications)
            self.observation_space = DataSpace(self.contexts)

            # ------------- Create the engagements using the classifier ------- #
            self.engagements = self.predictEngagements(self.notifications, self.contexts)
            self.engagements = pd.DataFrame(self.engagements, columns=['action'])
            
        
        self.epoch = 0
        self.openedNotifications = []
        self.openedActions = []
        self.openedContexts = []
        self.dismissedNotifications = []
        self.dismissedActions = []
        self.dismissedContexts = []
        self.correctlyOpened = 0
        self.correctlyDismissed = 0
        
        if test:
            return self.contexts
        else:
            return self.contexts, self.notifications, self.engagements
        
    def evaluate(self, notifications=None):
        
        if isinstance(notifications, pd.DataFrame):
            if len(notifications) == len(self.contexts):
                self.notifications = notifications
            else: 
                print('Incorrect number of notifications submitted')
                return
            
        results = self.execute_evaluation(self.notifications, self.contexts, self.test)
        print(results)
            
        
    
class DataSpace:
    
    def __init__(self, data):
        self.info = {}
        for col in data:
            col_info = {'type': str(data[col].dtype)}
            if data[col].dtype == 'object':
                col_info['labels'] = LabelEncoder().fit(data[col])
            elif data[col].dtype == 'bool':
                col_info['labels'] = [True, False]
            else:
                col_info['max'] = data[col].max()
                col_info['min'] = data[col].min()
                col_info['mean'] = data[col].mean()
                col_info['median'] = data[col].median()
            self.info[col] = col_info
            
    def sample(self):
        sample = {}
        for col in self.info:
            if self.info[col]['type'] == 'object':
                sample[col] = choice(self.info[col]['labels'].classes_)
            elif self.info[col]['type'] == 'bool':
                sample[col] = choice([True, False])
            else:
                sample[col] = randrange(self.info[col]['min'], (self.info[col]['max']+1))
        return sample
        
class ClassificationBattery:
    
    def __init__(self, data, data_set_type, notif_cat, context_cat, context_scale, dir_path):
        
        
        if data_set_type is 'train':
            self.data_time_period = '3months'
        elif data_set_type is 'test': 
            self.data_time_period = '6months'
        else:
            self.data_time_period = '3months'
        
        # load the adaboost model as UI user
        self.clf = joblib.load(self.dir_path+'/data/user_1/'+self.data_time_period+'_Adaboost.joblib')
        self.data.action = self.prediction(self.data[self.notif_cat], self.data[self.context_cat + self.context_scale])
        
        self.results = []
                                          
    
    
    
    
    
        
    
      