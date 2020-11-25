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
import tf
import random
import torch
import scipy.misc as m


from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
bridge = CvBridge()


class CarlaEnv1():
    def __init__(self, index, bottom_policy):
        self.bottom_policy = bottom_policy
        self.state3 = np.zeros((512, 3))
        self.image_state = np.zeros((88, 200, 3))

        self.img = None
        self.img2 = None
        self.img3 = None

        self.is_crashed = None
        self.crossed_lane_markings = None
        self.speed = None
        self.index = index
        self.car_id = None
        self.lane = None
        self.command = None       #0:follow lane  1:turn left  2:go straight 3:turn right
        self.goal_point = [1, 222]

        colors = [[128, 64, 128],
                  [232, 35, 244], ]
        colors1 = [[0, 0, 0],
                  [159, 255, 84], ]
        self.label_colours = dict(zip(range(2), colors))
        self.label_colours1 = dict(zip(range(2), colors1))
        self.valid_classes = [0, 70, 190, 250, 220, 153, 157, 128, 244, 107, 102]
        self.our_classes = [128, 157, 244]  # road    road_line     other

        node_name = 'CarlaEnv_' + str(index)
        rospy.init_node(node_name, anonymous=None)
        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        

        image_state_topic = '/camera/central'  # front
        self.image_state_sub = rospy.Subscriber(image_state_topic, Image, self.image_callback)

        image_state_topic2 = '/camera/left'  # left
        self.image_state_sub2 = rospy.Subscriber(image_state_topic2, Image, self.image_callback2)

        image_state_topic3 = '/camera/right'  # left
        self.image_state_sub3 = rospy.Subscriber(image_state_topic3, Image, self.image_callback3)

        odom_topic = '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        laser_topic = '/scan'
        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        command_topic = '/joy'
        self.command_pub = rospy.Subscriber(command_topic,Joy, self.command_callback)

        self.sim_clock = rospy.Subscriber('/clock', Clock, self.sim_clock_callback)

        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None

        while self.command is None:
            pass

        rospy.sleep(1)

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

    def get_bottom_pixel(self):  # seg = visual_policy(rgb_list)
        cimg = copy.deepcopy(self.img)
        limg = copy.deepcopy(self.img2)
        rimg = copy.deepcopy(self.img3)
        cimg = cimg[:, :, 0:3]
        cimg = np.array(cimg, dtype=np.uint8)
        cimg = m.imresize(cimg, (128, 160))
        cimg = cimg.astype(np.float64)
        cimg = cimg.astype(float) / 255.0

        limg = limg[:, :, 0:3]
        limg = np.array(limg, dtype=np.uint8)
        limg = m.imresize(limg, (128, 160))
        limg = limg.astype(np.float64)
        limg = limg.astype(float) / 255.0

        rimg = rimg[:, :, 0:3]
        rimg = np.array(rimg, dtype=np.uint8)
        rimg = m.imresize(rimg, (128, 160))
        rimg = rimg.astype(np.float64)
        rimg = rimg.astype(float) / 255.0

        outc = self.bottom_policy.predict([cimg[None]])
        outl = self.bottom_policy.predict([limg[None]])
        outr = self.bottom_policy.predict([rimg[None]])

        bottom_c = np.argmax(outc[0][0], axis=1).astype(float)
        bottom_l = np.argmax(outl[0][0], axis=1).astype(float)
        bottom_r = np.argmax(outr[0][0], axis=1).astype(float)

        # bottom = bottom_l + bottom_c + bottom_r

        # print(len(bottom))
        bottom_l = (bottom_l / 127).reshape(1, -1)
        bottom_c = (bottom_c / 127).reshape(1, -1)
        bottom_r = (bottom_r / 127).reshape(1, -1)

        return bottom_l, bottom_c, bottom_r

    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)

    def odometry_callback(self, odometry):
        Quaternious = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.odom_location = [odometry.pose.pose.position.x, -odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def command_callback(self, data):
        if data.buttons[0] == 1:
            self.command = [1,0,0,0]
            print('***************************')
            print('follow lane')
            print('***************************')

        if data.buttons[2] == 1:
            self.command = [0,1,0,0]
            print('***************************')
            print('turn left')
            print('***************************')

        if data.buttons[3] == 1:
            self.command = [0,0,0,1]
            print('***************************')
            print('go straight')
            print('***************************')

        if data.buttons[1] == 1:
            self.command = [0,0,1,0]
            print('***************************')
            print('turn right')
            print('***************************')

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def get_image_observation(self, count):
        img = copy.deepcopy(self.img)
        # image_name = './picture/'+str(count) + '.png'
        # cv2.imwrite(image_name, img)
        img = img[:, :, 0:3]
        img = np.array(img, dtype=np.uint8)
        img = m.imresize(img, (84, 84))
        img = img.astype(np.float64)
        img = img.astype(float) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 6
        scan[np.isinf(scan)] = 6
        scan=np.array(scan)
        scan[scan>6]=6
        #scan[scan<0.01]=3
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_mum
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)

        return scan_sparse/6.0 - 0.5

    def get_self_speed(self):
        return self.speed

    def get_self_odom_location(self):
        return self.odom_location

    def get_command(self):
        return self.command

    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0] *0.7   # 0.7 action[0] * 2
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]*0.3    # 0.3
        self.cmd_vel.publish(move_cmd)


if __name__ == '__main__':
    env = CarlaEnv1(0, visual_policy)
    # env.get_image_observation_rgb2seg()
    i = 0
    j = 25
    env.reset_pose(j)
    while True:
        print(i)
        b = env.get_laser_observation()
        obs_seg = env.get_segmentation_observation(i)
        time.sleep(0.2)
        i += 1
        if i % 60 == 0:
            j += 1

