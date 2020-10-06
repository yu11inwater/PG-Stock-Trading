#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:28:07 2020

@author: chenyuchen
"""

import numpy as np
import market_model_builder
import math

class PolicyGradient:
    
    def __init__(self, env, gamma = 0.99, weights_path=None, file_name = "new_pg.h5"):
        
        self.env = env
        self.gamma = gamma
        self.weights_path = weights_path
        self.file_name = file_name
        self.model = market_model_builder.MarketPolicyGradientModelBuilder(weights_path).getModel()
        self.model.compile(loss='mse', optimizer='rmsprop')
    

    def train(self, max_episode = 20, max_path_length = 200):
        
        env = self.env
        model = self.model
        act = [-1,0,1]
        
        for e in range(max_episode):
            env.reset()
            observation = env.reset()
            game_over = False
            
            inputs = []
            outputs = []
            predicteds = []
            rewards = []
            infos_=[]
            
            #if not game over
            while not game_over:
                #get action prob 
                observation = np.array(observation)
                observation = observation.reshape(1,101)                
                inputs.append(observation)
                
                aprob = model.predict(observation)[0]
                predicteds.append(aprob)
                
                #choose action
                act = [-1,0,1]
               
                #if overweight, choose action from limited action space
                if env.position_record != []:
                    pos = env.position_record[len(env.position_record)-1]
                    if pos == 1:
                        act = [-1,0]
                        aprob = aprob[:2]
                    elif pos == -1:
                        act = [0,1]
                        aprob = aprob[1:]
                   
                
                
                #-----if aprob = [0,0]
                '''
                if np.sum(aprob) == 0:
                    print(aprob)
                    action = np.random.choice(act, 1, p=None)[0]
                    print(action)
                    #input()
                else:
                    #choose action based on the probability distribution
                    action = np.random.choice(act, 1, p=aprob/np.sum(aprob))[0]
                    '''
                #------
                
                #choose action based on the probability distribution
                action = np.random.choice(act, 1, p=aprob/np.sum(aprob))[0]
                
                
                observation, reward, game_over, info = env.step(action)
                print('reward = ',reward)
               
                #redistributed aprob, for recording
                if len(aprob) != 3:
                    a1 = math.exp(aprob[0])/(math.exp(aprob[1])+math.exp(aprob[0]))
                    a2 = math.exp(aprob[1])/(math.exp(aprob[1])+math.exp(aprob[0]))
                    if pos == 1:
                        aprob = [a1,a2,0]
                    elif pos == -1:
                        aprob = [0,a1,a2]
                print('----\naprob = ',aprob)
                print('action = ',action)
                outputs.append(np.array(aprob))
                rewards.append(float(reward))
                infos_.append(info)
            
          
            inputs = inputs[100:]
            outputs = outputs[100:]
            predicteds = predicteds[100:]
            rewards = rewards[100:]
            
          
            inputs_ = np.vstack(inputs)
            outputs_ = np.array(outputs)
            predicteds_ = np.array(predicteds)
            rewards_ = np.array(rewards)


            
            i=0
            for reward in rewards:
                #outputs_[i] = 0.5 + (2 * outputs_[i] - 1) * discounted_reward
                if reward < 0:
                    outputs_[i] = 1 - outputs_[i]
                    outputs_[i] = outputs_[i] / sum(outputs_[i])

                outputs_[i] = np.minimum(1, np.maximum(0, predicteds_[i] + (outputs_[i] - predicteds_[i]) * abs(reward)))
              
                i += 1
                if i == outputs_.shape[0]:
                    break
                
                
            model.fit(inputs_, outputs_, epochs = 1, verbose = 0, shuffle = True)
            model.save_weights(self.file_name)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            