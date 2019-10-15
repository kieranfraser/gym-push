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

class EvalUMAP2(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        print('EvalUMAP2-v0 environment')
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        
        try:
            os.makedirs(self.dir_path+'/results/train/task2/')
        except FileExistsError:
            pass
        try:
            os.makedirs(self.dir_path+'/results/test/task2/')
        except FileExistsError:
            pass
        
        # ------------ load data ------------------ #
        # ------------ train as default ------------------ #
        self.data = pd.read_csv(self.dir_path+'/data/user_1/train_set.csv' )
        self.data = self.data.sort_values(by=['time'])
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
        
        # not always based on the train data
        self.max_diversity = self.data[self.notif_cat].nunique().sum()
        
        # ------------- Create the engagements using the classifier ------- #
        self.models = ['Adaboost',
                       # 'Random Forest', 
                       # 'Nearest Neighbors',
                       'Decision Tree',
                       'Naive Bayes']
        
        self.engagements = self.predictEngagements(self.notifications, self.contexts)
        self.engagements = pd.DataFrame(self.engagements, columns=['action'])
        
        self.epoch = 0
        self.accumulated_notifications = pd.DataFrame()
        self.diversity_results = [{'epoch': 0, 'score': 0}]
        self.ctr_results = self.initial_ctr()
        self.enticement_results = [{'epoch': 0, 'score': 0}]
        
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
        
        
        # ensures plots are only shown when called and open
        plt.ioff() 
    
    '''
        Evaluates the action at the epoch. Steps through the contexts by incrementing the epoch. 
        Returns the next context observed, the reward from the action in the previous context,
        whether or not the environment is finished stepping through observations and an info object.
        In this scenario, the action == notification, the observation == context, the reward == engagement.
    '''
    def step(self, notification):
        if isinstance(notification, pd.DataFrame):
            engagement = self.predictEngagements(notification, self.contexts.iloc[[self.epoch]])
            engagement = pd.DataFrame(engagement, columns=['action'])
            
            self.accumulated_notifications = self.accumulated_notifications.append(notification)

            # Calculate performance metrics for completed epoch before moving on
            
            if self.epoch % self.verbosity == 0 and self.epoch > 0:
                info = self.execute_evaluation(self.accumulated_notifications,
                                           self.contexts[:len(self.accumulated_notifications)],
                                           self.test, False)
            else:
                info = {}

            done = False
            self.epoch = self.epoch + 1
            if self.epoch == (len(self.contexts)):
                done = True
                context = pd.DataFrame(columns=(self.context_cat+self.context_scale))
                info = self.execute_evaluation(self.accumulated_notifications,
                                           self.contexts[:len(self.accumulated_notifications)],
                                           self.test, True)
            else:
                context = self.contexts.iloc[[self.epoch]]
            
            return context, engagement, done, info
        else:
            print('notifications must be passed as a DataFrame with column names identical'+
                  'to those of the training data (also described in the docs)')
        
    
    '''
        Resets the environment to default values (default data is training data)
    '''
    def reset(self, test=False, verbosity=1000):
        # ------------ load data ------------------ #
        # ------------ train as default ------------------ #
        print('Resetting environment.')
        if test:
            self.test = True
            self.data = pd.read_csv(self.dir_path+'/data/user_5/test_set.csv' )
            self.data = self.data.sort_values(by=['time'])
            self.clf = joblib.load(self.dir_path+'/data/user_5/6months_Adaboost.joblib')
            
            self.notif_ohe = joblib.load(self.dir_path+'/data/user_5/notif_ohe.joblib')
            self.context_ohe = joblib.load(self.dir_path+'/data/user_5/context_ohe.joblib')
            self.context_scaler = joblib.load(self.dir_path+'/data/user_5/context_scaler.joblib')

            self.notif_cat = ['appPackage', 'category', 'ledARGB', 'priority', 'vibrate', 'visibility', 'subject', 'enticement', 'sentiment']
            self.context_cat = ['timeAppLastUsed', 'timeOfDay', 'dayOfWeek']
            self.context_scale = ['unlockCount_prev2', 'uniqueAppsLaunched_prev2', 'dayOfMonth']

            self.contexts = self.data[(self.context_cat+self.context_scale)]
            self.notifications = self.data[self.notif_cat]
            self.action_space = DataSpace(self.notifications)
            self.observation_space = DataSpace(self.contexts)

            # ------------- Create the engagements using the classifier ------- #
            self.engagements = self.predictEngagements(self.notifications, self.contexts)
            self.engagements = pd.DataFrame(self.engagements, columns=['action'])
            self.max_diversity = self.data[self.notif_cat].nunique().sum()
            
        else:
            self.test = False
            self.data = pd.read_csv(self.dir_path+'/data/user_1/train_set.csv' )
            self.data = self.data.sort_values(by=['time'])
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

            # ------------- Create the engagements using the classifier ------- #
            self.engagements = self.predictEngagements(self.notifications, self.contexts)
            self.engagements = pd.DataFrame(self.engagements, columns=['action'])
            self.max_diversity = self.data[self.notif_cat].nunique().sum()
            
        self.verbosity = verbosity
        self.epoch = 0
        self.accumulated_notifications = pd.DataFrame()
        self.diversity_results = [{'epoch': 0, 'score': 0}]
        self.ctr_results = self.initial_ctr()
        self.enticement_results = [{'epoch': 0, 'score': 0}]
        
        
        self.openedNotifications = []
        self.openedActions = []
        self.openedContexts = []
        self.dismissedNotifications = []
        self.dismissedActions = []
        self.dismissedContexts = []
        self.correctlyOpened = 0
        self.correctlyDismissed = 0
        
        return self.contexts.iloc[[self.epoch]]
    
    '''
        Shuts down the environment
    '''
    def close(self):
        print('To do.. close')
        
    # ----------- Metric methods ----------#
    
    '''
        Helper method for donut charts - ensures just the relevant metric
        is printed. 
    '''
    def autopct_generator(self, ignored):
        def inner_autopct(pct):
            if round(pct, 2)==ignored:
                return ''
            else:
                return ('%.2f' % pct)+'%'
        return inner_autopct
    
    def initial_ctr(self):
        tmp = []
        for name in self.models:
            tmp.append({'epoch': 0, 'model': name, 'ctr_score': 0})
        return tmp        
    
    '''
        Saves the results of the evaluation and generates charts. Arguments are the 
        different metric results from the evaluation and the data_set_type used in 
        the evaluation (train or test).
    '''
    def save_results(self, data_set_type, finished):
        
        sns.set(style='white', font_scale=1.5, palette='pastel')
        tmp = pd.DataFrame(self.ctr_results)
        ax = sns.catplot(x="epoch", y="ctr_score", hue="model", 
                         kind="point", data=tmp, height=8, aspect=1, ci=None)
        ax.set(xlabel='Epoch', ylabel='CTR (%)', title='Click-Through-Rate over Time')
        plt.savefig(self.dir_path+'/results/'+data_set_type+'/task2/ctr_results.png', bbox_inches='tight')

        tmp = pd.DataFrame(self.diversity_results)
        ax = sns.catplot(x="epoch", y="score", 
                         kind="point", data=tmp, height=7, aspect=1.2, ci=None)
        ax.set(xlabel='Epoch', ylabel='Score (%)', title='Diversity over Time')
        plt.savefig(self.dir_path+'/results/'+data_set_type+'/task2/diversity_results.png', bbox_inches='tight')

        tmp = pd.DataFrame(self.enticement_results)
        ax = sns.catplot(x="epoch", y="score", 
                         kind="point", data=tmp, height=7, aspect=1.2, ci=None)
        ax.set(xlabel='Epoch', ylabel='Score (%)', title='Enticement over Time')
        plt.savefig(self.dir_path+'/results/'+data_set_type+'/task2/enticement_results.png', bbox_inches='tight')
        
        if not finished:
            print('Updated results at epoch: ', self.epoch)
            plt.close("all")
        else:
            print('Finished. Saving final results at epoch: ', self.epoch)
            # --------- Model CTR Bar (not shown, but saved) ---------------- #
            tmp = pd.DataFrame(self.ctr_results[-len(self.models):])
            ax = sns.catplot(x="model", y="ctr_score", data=tmp, height=6, kind="bar")
            ax.set(xlabel='Model', ylabel='CTR (%)', title="CTR performance of notifications")
            plt.savefig(self.dir_path+'/results/'+data_set_type+'/task2/ctr_final_bar.png', bbox_inches='tight')
            plt.clf()


            # --------- AdaBoost CTR Donut ---------------- #
            adaboost_score = self.ctr_results[-len(self.models):][0]['ctr_score']
            if round(adaboost_score, 2) == 50.00:
                adaboost_score = 50.01
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))
            size=[round(adaboost_score,2), round(np.absolute(100-adaboost_score),2)]
            my_circle=plt.Circle( (0,0), 0.7, color='white')
            plt.pie(size, autopct=self.autopct_generator(round(np.absolute(100-adaboost_score),2)),
                    pctdistance=0.01, colors=['aquamarine','peachpuff'])
            ax.set_title("Click-Through-Rate Score", fontsize= 22)
            plt.rcParams.update({'font.size': 22})
            p=plt.gcf()
            p.gca().add_artist(my_circle)
            plt.savefig(self.dir_path+'/results/'+data_set_type+'/task2/ctr_final_donut.png', bbox_inches='tight')

            # --------- Enticement Donut ---------------- #
            enticement_score = self.enticement_results[-1]['score']
            if round(enticement_score, 2) == 50.00:
                enticement_score = 49.99
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))
            size=[round(enticement_score,2), round(np.absolute(100-enticement_score),2)]
            my_circle=plt.Circle( (0,0), 0.7, color='white')
            plt.pie(size, autopct=self.autopct_generator(round(np.absolute(100-enticement_score),2)),
                    pctdistance=0.01, colors=['peachpuff','aquamarine'])
            ax.set_title("Enticement Score", fontsize= 22)
            plt.rcParams.update({'font.size': 22})
            p=plt.gcf()
            p.gca().add_artist(my_circle)
            plt.savefig(self.dir_path+'/results/'+data_set_type+'/task2/enticement_final_donut.png', bbox_inches='tight')

            # --------- Diversity Donut ---------------- #
            diversity_score = self.diversity_results[-1]['score']
            if round(diversity_score, 2) == 50.00:
                diversity_score = 50.01
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))
            size=[round(diversity_score,2), round(np.absolute(100-diversity_score),2)]
            my_circle=plt.Circle( (0,0), 0.7, color='white')
            plt.pie(size, autopct=self.autopct_generator(round(np.absolute(100-diversity_score),2)),
                    pctdistance=0.01, colors=['aquamarine','peachpuff'])
            ax.set_title("Diversity Score", fontsize= 22)
            plt.rcParams.update({'font.size': 22})
            p=plt.gcf()
            p.gca().add_artist(my_circle)
            plt.savefig(self.dir_path+'/results/'+data_set_type+'/task2/diversity_final_donut.png', bbox_inches='tight')

            joblib.dump(self.ctr_results, self.dir_path+'/results/'+data_set_type+'/task2/ctr_results.joblib')
            joblib.dump(self.enticement_results, self.dir_path+'/results/'+data_set_type+'/task2/enticement_results.joblib')
            joblib.dump(self.diversity_results, self.dir_path+'/results/'+data_set_type+'/task2/diversity_results.joblib')
            
            plt.show()
        
        print('Results saved here: ', self.dir_path+'/results/'+data_set_type+'/task2/')
        
    '''
        Convert enticement string to tiered value
    '''
    def enticement_to_value(self, val):
        if val == 'high':
            return 3
        elif val == 'moderate':
            return 2
        else:
            return 1
    
    '''
        Calculates the metrics (CTR, Diversity, Enticement) for the generated
        notifications. Arguments are the notifications, contexts and indication
        of train or test evaluation to be performed. Returns the results in dict.
    '''
    def execute_evaluation(self, notifications, contexts, test, finished):
        if test:
            data_time_period = '6months'
            user = 'user_5'
        else:
            data_time_period = '3months'
            user = 'user_1'
            
        # Metrics based on end-game potential - will show how we approach this value 
        # as we move through epochs
        # Should both start relatively close to 0 and grow as move through contexts
        diversity_score = (notifications.nunique().sum()/self.max_diversity)*100
        self.diversity_results.append({'epoch': self.epoch, 'score': diversity_score})
        
        a = len(contexts)
        b = ((a*3)-a)
        enticement_score = ((notifications.enticement.apply(self.enticement_to_value).sum() - a)/b)*100
        self.enticement_results.append({'epoch': self.epoch, 'score': enticement_score})
            
        X = self.encode(notifications, contexts)
        
        for name in self.models:
            
            model = joblib.load(self.dir_path+'/data/'+user+'/'+data_time_period+'_'+name+'.joblib')
            predictions = model.predict(X)
            self.ctr_results.append({'epoch': self.epoch, 'model': name, 'ctr_score': ((np.sum(predictions)/len(predictions))*100)})
           
        self.save_results('test' if test else 'train', finished)
        return {'ctr':self.ctr_results, 'diversity':self.diversity_results, 'enticement':self.enticement_results}

    # ----------- Simulated user methods ----------#
    '''
        Encodes a notification, context pair and concatenates ready for use by 
        classifiers.
    '''
    def encode(self, notifications, contexts):
        notif_enc = self.notif_ohe.transform(notifications) 
        context_enc_cat = self.context_ohe.transform(contexts[self.context_cat])
        context_enc_scale = self.context_scaler.transform(contexts[self.context_scale]) 
        return np.concatenate([notif_enc, context_enc_cat, context_enc_scale], axis=1)
    
    '''
        Takes notification, context pairs and predicts whether or not the user
        will open them. Returns actions (1 for opened, 0 for dismissed)
    '''
    def predictEngagements(self, notifications, contexts):
        enc = self.encode(notifications, contexts)
        return self.clf.predict(enc)
    
    # ----------- EvalUMAP methods ----------#
    
    '''
        Called by the participant to receive data for the Task. Argument is a variable 
        indicating whether they wish to receive test or train data. Resets all the 
        necessary variables and returns data.
    '''
    def request_sample_data(self):
        return self.contexts.head(), self.notifications.head(), self.engagements.head()
            
        
'''
    Class which is used to describe the action and observation spaces.
    The action space is the possible notifications.
    The observation space is the possible contexts.
'''
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
            
    '''
        Takes a random sample from each column and returns in dict form.
    '''
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
                                          
    
    
    
    
    
        
    
      