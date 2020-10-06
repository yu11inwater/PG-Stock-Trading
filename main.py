#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:31:50 2020

@author: chenyuchen
"""
#%%

import configparser
import numpy as np
from Q_learning_epsilon_env import MarketEnv

#import market_model_builder
from pg import PolicyGradient
#import datetime
     
from sqlalchemy import create_engine
import pandas as pd


def get_data():
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    cfg = config['DB']
    engine = create_engine("mysql+pymysql://"+cfg['username']+":"+cfg['passwd']+"@"+cfg['host']+":"+cfg['port']+"/"+cfg['database']+"?charset=utf8")
    
    script='SELECT * FROM chips_db.txf20110101_20191231_60min'
    df = pd.read_sql(script, con=engine)
    #df.drop(columns=['index'], inplace=True)
    df = df.set_index('Date')
    return df

# 5ma data
def get_5MA_data(df):
    # calculate 5 point MA 
    df_5dMA = df['Close'].rolling(window=5).mean()
    df_5dMA[0] = 9012.4
    df_5dMA[1] = 9012.4
    df_5dMA[2] = 9012.4
    df_5dMA[3] = 9012.4  
    df['Close'] = df_5dMA

    return df


#%% data settings
df = get_data()

# training, testing data split
train_start = '2014-01-01'
train_test_split = '2015-01-01'
training_data = df[train_start:train_test_split]


train_test_split = '2015-01-01'

#train_test_split = '2016-01-01'
test_end = '2016-01-01'
testing_data = df[train_test_split:test_end]


#%% train
env = MarketEnv(training_data,60)
#pg = PolicyGradient(env, gamma= 0.9,file_name = "pg_3.h5")
pg = PolicyGradient(env, gamma= 0.9,file_name = "new34.h5")
pg.train()



#%% test
env_test = MarketEnv(testing_data,60)
pg_test = PolicyGradient(env_test, gamma= 0.9,weights_path = "new34.h5")
model = pg_test.model
act = [-1,0,1]

env_test.reset()
observation = env_test.reset()
game_over = False

inputs = []
outputs = []
predicteds = []
rewards = []
infos = []
actions =[]

#if not game over
while not game_over:
    #get action prob 
    observation = np.array(observation)
    observation = observation.reshape(1,61)                
    
    aprob = model.predict(observation)[0]
    inputs.append(observation)
    predicteds.append(aprob)
        
    #choose action
    act = [-1,0,1]

    #if overweight, choose action from limited action space
    if env_test.position_record != []:
        pos = env_test.position_record[len(env_test.position_record)-1]
        if pos == 1:
            act = [-1,0]
            aprob = aprob[:2]
        elif pos == -1:
            act = [0,1]
            aprob = aprob[1:]
    
    #print(aprob)
    #-----
    max_act = np.argmax(aprob)
    action = act[max_act]
    #print(action)
    actions.append(action)
    
    #outputs.append(np.array(y))
    observation, reward, game_over, info = env_test.step(action)
   
    rewards.append(float(reward))
  
# accumulated wealth of the whole trading procedure   
wealths = env_test.wealths
total_values = env_test.total_values
positions = env_test.position_record

import matplotlib.pyplot as plt
testing_data = testing_data.reset_index()
past_t = env_test.past_t
# performance visualization
xaxis = np.linspace(1, testing_data.shape[0] - past_t, testing_data.shape[0] - past_t)
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(wealths, "red")
plt.title("Accumulated wealth")
plt.xlabel("Time points")
plt.ylabel("Wealth")
plt.xticks(np.arange(1, testing_data.shape[0], step=1))
plt.subplot(2, 2, 2)
plt.plot(total_values, "blue")
plt.title("Total values")
plt.xlabel("Time points")
plt.ylabel("Values")
plt.xticks(np.arange(1, testing_data.shape[0], step=1))
plt.subplot(2, 2, 3)
plt.bar(xaxis, positions)
plt.scatter(xaxis, actions, c="b")
plt.title("Positions on hand")
plt.xlabel("Time points")
plt.ylabel("Positions")
plt.xticks(np.arange(1, testing_data.shape[0], step=1))
plt.subplot(2, 2, 4)
plt.plot(testing_data["Close"], "red")
plt.title("Index")
plt.xlabel("Time points")
plt.ylabel("Index")
plt.xticks(np.arange(1, testing_data.shape[0], step=1))

'''
plt.scatter(xaxis, actions, c="b")
plt.title("Actions")
plt.xlabel("Time points")
plt.ylabel("Action")
plt.xticks(np.arange(1, testing_data.shape[0], step=50))
'''
#plt.ylim(-1, 1)
plt.show()

print(len(positions))
print(len(actions))
print(len(xaxis))
print(env_test.past_t)









