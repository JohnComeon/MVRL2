#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" main script """
import random
import time
import rospy
import copy
import numpy as np
import cv2
import scipy
import sys
import math
import os
import logging
import socket
import matplotlib.pyplot as plt
import tf

import torch

import torch.nn as nn

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

setup_seed(1)


from keras.models import model_from_json
import keras.backend as K
import torch
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf2

np.set_printoptions(threshold=25000)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf2.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf2.Session(config=config)
ktf.set_session(session)

from env import CarlaEnv1
from torch.optim import Adam
from collections import deque
#from network.net import CNNPolicy
from ppo import generate_action_no_sampling
from ppo import transform_buffer
#from network.erfnet import erfnet
from vlnet import VLnet

MAX_EPISODES = 10000
LASER_BEAM = 512
HORIZON = 256
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 256
EPOCH = 4
COEFF_ENTROPY = 0.01
CLIP_VALUE = 0.1
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 3e-4
NUM_ENV = 1
alpha = 0.5
def run(env, vl_policy , policy_path, action_bound, optimizer):
    # rate = rospy.Rate(5)

    global_step = 0

    vl_policy.eval()
    #print(11111111111111111111111111111)
    for id in range(MAX_EPISODES):
        obs_bottom_left, obs_bottom_central, obs_bottom_right = env.get_bottom_pixel()
        #print(obs_bottom_left)
        command = env.get_command()
        speed = np.asarray(env.get_self_speed())
        if speed[0] < 0:
            speed[0] = 0

        state = [obs_bottom_left, obs_bottom_central, obs_bottom_right, speed, command]
        last_w = 0
        while not rospy.is_shutdown():
            start = time.time()
            state_list = []
            state_list.append(state)

            # generate actions at rank==0
            _,_,mean, scaled_action,scaled_action1 = generate_action_no_sampling(env=env, state_list=state_list, vl_policy=vl_policy, action_bound=action_bound)
            # execute actions
            print(scaled_action1[0])
            #scaled_action1[0][1] = (1 - alpha) * last_w  + alpha * scaled_action1[0][1]
            env.control_vel(scaled_action1[0])
            last_w = scaled_action1[0][1]

            time.sleep(0.15)
            #env.control_vel(scaled_action[0])
            #print(scaled_action1[0])
            global_step+=1
            # get next state

            obs_bottom_left, obs_bottom_central, obs_bottom_right = env.get_bottom_pixel()
 
            speed_next = np.asarray(env.get_self_speed())
            #last_w = speed_next[1]
            if speed[0] < 0:
                speed[0] = 0
            commond_next = env.get_command()

            state_next = [obs_bottom_left, obs_bottom_central, obs_bottom_right, speed_next, commond_next]

            state = state_next
            end = time.time()
            #print (end -start)


if __name__ == '__main__':
    reward = None
    action_bound = [[0, -1], [1, 1]]

    vl_policy_path = '/home/nvidia/MVRL_Outdoor/multicamerarl/bottom_policy_models' # 'vl_models_no_car_and_person_best'-->1360       'vl_models'-->1200

    vl_policy = VLnet(action_space=2)
    torch.cuda.current_device()
    vl_policy.cuda()

    opt = Adam(vl_policy.parameters(), lr=LEARNING_RATE)

    vl_file = vl_policy_path + '/stage1_2460 episode_2516'    #'/Stage1_7500'   #1460  1200 1040 660 620 600 560 500

    def relu6(x):
        """Relu 6
        """
        return K.relu(x, max_value=6.0)

    def hard_swish(x):
        return x*K.relu(x+3.0, max_value=6.0)/6.0

    def jsonToModel(json_model_path):
        with open(json_model_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json, custom_objects={'tf': tf2, 'relu6': relu6,'hard_swish': hard_swish})
        return model

    model_pixel = '/home/nvidia/MVRL_Outdoor/multicamerarl/model/model_struct0828.json'
    pixel_weights = '/home/nvidia/MVRL_Outdoor/multicamerarl/model/weights_040.h5'
    bottom_policy = jsonToModel(model_pixel)
    bottom_policy.load_weights(pixel_weights)


    if os.path.exists(vl_file):
        print('####################################')
        print('############Loading_VLModel###########')
        print('####################################')
        state_dict = torch.load(vl_file)
        vl_policy.load_state_dict(state_dict)

        #print(1111111111111111111)
    else:
        print('#####################################')
        print('############Start Training###########')
        print('#####################################')

    env = CarlaEnv1(index=0, bottom_policy=bottom_policy)

    try:
        run(env=env, vl_policy = vl_policy, policy_path=vl_policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass



