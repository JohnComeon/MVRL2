#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" main script """

import time
import rospy
import copy
import numpy as np
import cv2
import sys
import math
import os
import logging
import socket
import matplotlib.pyplot as plt
import tf

import random
from mpi4py import MPI
from torch.optim import Adam
from torch.autograd import Variable
from collections import deque
import torch
import torch.nn as nn
import scipy.misc as m

from keras.models import model_from_json
import keras.backend as K
import tensorflow as tf2

#sys.path.append("/home/hxq/multi-navigation/planner")
#from planner import Planner

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()


class bottom_pub():
    def __init__(self, bottom_policy):
        self.bottom_policy = bottom_policy
        self.img = None
        self.img2 = None
        self.img3 = None

        node_name = 'bottom'
        rospy.init_node(node_name, anonymous=None)

        frontbc_topic = '/frontbc'
        self.bottom_pub = rospy.Publisher(frontbc_topic, Image, queue_size=10)
        leftbc_topic = '/leftbc'
        self.bottom_pub2 = rospy.Publisher(leftbc_topic, Image, queue_size=10)
        rightbc_topic = '/rightbc'
        self.bottom_pub3 = rospy.Publisher(rightbc_topic, Image, queue_size=10)

        image_state_topic = '/camera/central'  # front
        self.image_state_sub = rospy.Subscriber(image_state_topic, Image, self.image_callback)

        image_state_topic2 = '/camera/left'  # left
        self.image_state_sub2 = rospy.Subscriber(image_state_topic2, Image, self.image_callback2)

        image_state_topic3 = '/camera/right'  # right
        self.image_state_sub3 = rospy.Subscriber(image_state_topic3, Image, self.image_callback3)

        rospy.sleep(0.5)

    def image_callback(self, image_data):
        try:
            image_type = image_data.encoding
            image_data = bridge.imgmsg_to_cv2(image_data, image_type)
        except CvBridgeError, e:
            print
            e
        self.img = np.asarray(image_data, dtype=np.float32)
    def image_callback2(self, image_data):

        try:
            image_type = image_data.encoding
            image_data = bridge.imgmsg_to_cv2(image_data, image_type)
        except CvBridgeError, e:
            print
            e
        self.img2 = np.asarray(image_data, dtype=np.float32)
    def image_callback3(self, image_data):
        try:
            image_type = image_data.encoding
            image_data = bridge.imgmsg_to_cv2(image_data, image_type)
        except CvBridgeError, e:
            print
            e
        self.img3 = np.asarray(image_data, dtype=np.float32)

    def bottom_observation(self):
        cimg = copy.deepcopy(self.img)
        cimg = cimg[:, :, 0:3]
        cimg = np.array(cimg, dtype=np.uint8)
        cimg = m.imresize(cimg, (128, 160))
        cimg = cimg.astype(np.float64)
        cimg = cimg.astype(float) / 255.0

        limg = copy.deepcopy(self.img2)
        limg = limg[:, :, 0:3]
        limg = np.array(limg, dtype=np.uint8)
        limg = m.imresize(limg, (128, 160))
        limg = limg.astype(np.float64)
        limg = limg.astype(float) / 255.0

        rimg = copy.deepcopy(self.img3)
        rimg = rimg[:, :, 0:3]
        rimg = np.array(rimg, dtype=np.uint8)
        rimg = m.imresize(rimg, (128, 160))
        rimg = rimg.astype(np.float64)
        rimg = rimg.astype(float) / 255.0

        img = copy.deepcopy(self.img)
        img = img[:, :, 0:3]
        # cv2.imshow('221111',img)
        img = np.array(img, dtype=np.uint8)
        img = m.imresize(img, (128, 160))

        img2 = copy.deepcopy(self.img2)
        img2 = img2[:, :, 0:3]
        img2 = np.array(img2, dtype=np.uint8)
        img2 = m.imresize(img2, (128, 160))

        img3 = copy.deepcopy(self.img3)
        img3 = img3[:, :, 0:3]
        img3 = np.array(img3, dtype=np.uint8)
        img3 = m.imresize(img3, (128, 160))

        outc = self.bottom_policy.predict([cimg[None]])
        outl = self.bottom_policy.predict([limg[None]])
        outr = self.bottom_policy.predict([rimg[None]])
        bottom_c = np.argmax(outc[0][0], axis=1)
        bottom_c = bottom_c.tolist()
        bottom_l = np.argmax(outl[0][0], axis=1)
        bottom_l = bottom_l.tolist()
        bottom_r = np.argmax(outr[0][0], axis=1)
        bottom_r = bottom_r.tolist()

        img_bg = np.ones([128, 160, 3], np.uint8)
        img_bg = img_bg * 255
        img_bg2 = np.ones([128, 160, 3], np.uint8)
        img_bg2 = img_bg2 * 255
        img_bg3 = np.ones([128, 160, 3], np.uint8)
        img_bg3 = img_bg3 * 255
        # cv2.imshow('1111',img_bg)
        # cv2.waitKey(900)

        for i in range(len(bottom_c)):
            ptStart = (i, 240)
            ptEnd = (i, int(bottom_c[i]))
            green = (0, 255, 0)
            cv2.line(img_bg, ptStart, ptEnd, green, 1)
        for i in range(len(bottom_l)):
            ptStart = (i, 240)
            ptEnd = (i, int(bottom_l[i]))
            green = (0, 255, 0)
            cv2.line(img_bg2, ptStart, ptEnd, green, 1)
        for i in range(len(bottom_r)):
            ptStart = (i, 240)
            ptEnd = (i, int(bottom_r[i]))
            green = (0, 255, 0)
            cv2.line(img_bg3, ptStart, ptEnd, green, 1)

        img = cv2.addWeighted(img, 0.8, img_bg, 0.2, 0)
        img2 = cv2.addWeighted(img2, 0.8, img_bg2, 0.2, 0)
        img3 = cv2.addWeighted(img3, 0.8, img_bg3, 0.2, 0)
        # print(type(img))
        try:
            img = bridge.cv2_to_imgmsg(img, 'bgr8')
            img2 = bridge.cv2_to_imgmsg(img2, 'bgr8')
            img3 = bridge.cv2_to_imgmsg(img3, 'bgr8')
        except CvBridgeError, e:
            print
            e
        # cv2.imshow('1111',img)
        # cv2.waitKey(900)
        print('success....')

        self.bottom_pub.publish(img)
        self.bottom_pub2.publish(img2)
        self.bottom_pub3.publish(img3)


if __name__ == '__main__':

    def relu6(x):
        """Relu 6
        """
        return K.relu(x, max_value=6.0)


    def hard_swish(x):
        return x*K.relu(x+3.0, max_value=6.0)/6.0

    def jsonToModel(json_model_path):
        with open(json_model_path, 'r') as json_file:
            loaded_model_json = json_file.read()
            # print(loaded_model_json)
        model = model_from_json(loaded_model_json, custom_objects={'tf': tf2, 'relu6': relu6,'hard_swish': hard_swish})
        return model


    model_pixel = '/home/nvidia/multicamerarl/model/model_struct0828.json'
    pixel_weights = '/home/nvidia/multicamerarl/model/weights_040.h5'
    bottom_policy = jsonToModel(model_pixel)
    bottom_policy.load_weights(pixel_weights)
    while True:
        pub = bottom_pub(bottom_policy=bottom_policy)
        pub.bottom_observation()
        time.sleep(0.1)
