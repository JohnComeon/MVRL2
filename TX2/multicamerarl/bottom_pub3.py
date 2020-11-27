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
from torchvision import transforms
import scipy.misc as m


from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
import borderlinenet
trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

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
        cimg = cimg.astype(np.uint8)
        cimg = cimg[:, :, 0:3]
        cimg = cv2.resize(cimg, (160, 128))
        cimg = trans(cimg)
        cimg = cimg.unsqueeze(0)
        cimg = cimg.cuda()

        limg = copy.deepcopy(self.img2)
        limg = limg.astype(np.uint8)
        limg = limg[:, :, 0:3]
        limg = cv2.resize(limg, (160, 128))
        limg = trans(limg)
        limg = limg.unsqueeze(0)
        limg = limg.cuda()

        rimg = copy.deepcopy(self.img3)
        rimg = rimg.astype(np.uint8)
        rimg = rimg[:, :, 0:3]
        rimg = cv2.resize(rimg, (160, 128))
        rimg = trans(rimg)
        rimg = rimg.unsqueeze(0)
        rimg = rimg.cuda()

        img = copy.deepcopy(self.img)
        img = img[:, :, 0:3]
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

        outc = self.bottom_policy(cimg)
        outl = self.bottom_policy(limg)
        outr = self.bottom_policy(rimg)
        bottom_c = outc.data.cpu().numpy()
        bottom_c = np.argmax(bottom_c[0], axis=1)
        bottom_l = outl.data.cpu().numpy()
        bottom_l = np.argmax(bottom_l[0], axis=1)
        bottom_r = outr.data.cpu().numpy()
        bottom_r = np.argmax(bottom_r[0], axis=1)

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
    def getModel(weights_path):
        '''
          Initialize model.
          ## Arguments
            `img_dims`: Target image dimensions.
            `img_channels`: Target image channels.
            `output_dim`: Dimension of model output.
            `weights_path`: Path to pre-trained model.
          ## Returns
            `model`: the pytorch model
        '''
        model = borderlinenet.MobileNetV3_Small()
        # if weights path exists...
        if weights_path:
            try:
                model.load_state_dict(torch.load(weights_path))
                print("Loaded model from {}".format(weights_path))
            except:
                print("Impossible to find weight path. Returning untrained model")
        return model
    bottom_policy = getModel('./torch_models/best/weights_best.pth')

    while True:
        pub = bottom_pub(bottom_policy=bottom_policy)
        pub.bottom_observation()
        #time.sleep(0.1)
