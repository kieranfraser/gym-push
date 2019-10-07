import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pandas as pd
import numpy as np
import eel
from json_tricks import dumps

class Basic(gym.Env):
    metadata = {'render.modes': ['human']}
    
    @eel.expose
    def __init__(self):
        print('Basic-v0 environment')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # ------------ load data ------------------ #
        self.data = pd.read_csv(dir_path+'/data/notifications.csv' )
                
        self.contexts = self.data[['postedDayOfWeek', 'postedTimeOfDay', 'contactSignificantContext']]
        self.notifications = self.data[['appPackage', 'category', 'priority', 'numberUpdates']]
        self.engagements = self.data[['action', 'response']]
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(len(self.data))
        
        self.epoch = 0
        self.openedNotifications = []
        self.openedActions = []
        self.openedContexts = []
        self.dismissedNotifications = []
        self.dismissedActions = []
        self.dismissedContexts = []
        self.correctlyOpened = 0
        self.correctlyDismissed = 0
        
        # ------------ prep ui ------------------ #
        eel.init(dir_path+'/web')
        eel.start('main.html', mode='chrome', block=False)
        eel.sleep(3)
        eel.initial_state(dumps({'notification': self.notifications.iloc[self.epoch].to_dict(),
                          'context':self.contexts.iloc[self.epoch].to_dict(), 
                          'size':len(self.data)}))
        
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
        
    def calculate_reward(self, action):
        if self.epoch == 0: 
            ground_truth = self.engagements.iloc[self.epoch].action
            self.update_metrics(ground_truth, action)
            if ground_truth == action:
                return 1
            else:
                return -1
        else:
            if (self.epoch-1) >= 0: 
                ground_truth = self.engagements.iloc[(self.epoch-1)].action

                self.update_metrics(ground_truth, action)

                if ground_truth == action:
                    return 1
                else:
                    return -1
        return 0        
        
    def step(self, action):
        reward = self.calculate_reward(action)
        done = False
        self.epoch = self.epoch + 1
        if self.epoch == (len(self.data)-1):
            done = True
        info = {}
        
        observation = {**self.notifications.iloc[self.epoch].to_dict(), **self.contexts.iloc[self.epoch].to_dict()}
        
        return observation, reward, done, info
    
    def reset(self):
        
        self.epoch = 0
        self.openedNotifications = []
        self.dismissedNotifications = []
        self.correctlyOpened = 0
        self.correctlyDismissed = 0
        self.dismissedActions = []
        self.openedActions = []
        self.openedContexts = []
        self.dismissedContexts = []
        
        return {**self.notifications.iloc[self.epoch].to_dict(), **self.contexts.iloc[self.epoch].to_dict()}
    
    
    def render(self, mode='human'):
        eel.render(dumps({
                            'notification': self.notifications.iloc[self.epoch].to_dict(),
                            'context':self.contexts.iloc[self.epoch].to_dict(), 
                            'epoch':self.epoch,
                            'openedNotifications': self.openedNotifications,
                            'dismissedNotifications': self.dismissedNotifications,
                            'openedContexts': self.openedContexts,
                            'dismissedContexts': self.dismissedContexts,
                            'correctlyOpened': self.correctlyOpened,
                            'correctlyDismissed': self.correctlyDismissed
                        }))
    
    def close(self):
        print('close')
    