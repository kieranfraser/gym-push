import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pandas as pd
import numpy as np

import eel

class Basic(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print('init basic')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # ------------ load data ------------------ #
        self.data = pd.read_csv(dir_path+'/data/notifications.csv' )
        
        # ------------ prep ui ------------------ #
        
        eel.init(dir_path+'/web')
        eel.start('main.html', block=False)
        
        self.contexts = self.data[['postedDayOfWeek', 'postedTimeOfDay', 'contactSignificantContext']]
        self.notifications = self.data[['appPackage', 'category', 'priority', 'numberUpdates']]
        self.engagements = self.data[['action', 'response']]
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(len(self.data))
        
        self.epoch = 0
        
    def calculate_reward(self, action):
        if (self.epoch-1) >= 0: 
            ground_truth = self.engagements.iloc[(self.epoch-1)].action
            if ground_truth == action:
                return 1
            else:
                return -1
        return 0        
        
    def step(self, action):
        print('step')
        
        observation = np.concatenate([self.notifications.iloc[self.epoch].values, self.contexts.iloc[self.epoch].values])
        reward = self.calculate_reward(action)
        done = False
        self.epoch = self.epoch + 1
        if self.epoch == len(self.data):
            done = True
        info = {}
        
        return observation, reward, done, info
    
    def reset(self):
        print('reset')
        self.epoch = 0
    
    def render(self, mode='human'):
        print('render')
    
    def close(self):
        print('close')
    